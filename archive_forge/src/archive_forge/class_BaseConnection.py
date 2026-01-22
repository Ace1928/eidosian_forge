from __future__ import absolute_import
from __future__ import unicode_literals
import collections
import copy
import functools
import logging
import os
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import api_base_pb
from googlecloudsdk.third_party.appengine.api import apiproxy_rpc
from googlecloudsdk.third_party.appengine.api import apiproxy_stub_map
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.runtime import apiproxy_errors
class BaseConnection(object):
    """Datastore connection base class.

  NOTE: Do not instantiate this class; use Connection or
  TransactionalConnection instead.

  This is not a traditional database connection -- with App Engine, in
  the end the connection is always implicit in the process state.
  There is also no intent to be compatible with PEP 249 (Python's
  Database-API).  But it is a useful abstraction to have an explicit
  object that manages the database interaction, and especially
  transactions.  Other settings related to the App Engine datastore
  are also stored here (e.g. the RPC timeout).

  A similar class in the Java API to the App Engine datastore is
  DatastoreServiceConfig (but in Java, transaction state is always
  held by the current thread).

  To use transactions, call connection.new_transaction().  This
  returns a new connection (an instance of the TransactionalConnection
  subclass) which you should use for all operations in the
  transaction.

  This model supports multiple unrelated concurrent transactions (but
  not nested transactions as this concept is commonly understood in
  the relational database world).

  When the transaction is done, call .commit() or .rollback() on the
  transactional connection.  If .commit() returns False, the
  transaction failed and none of your operations made it to the
  datastore; if it returns True, all your operations were committed.
  The transactional connection cannot be used once .commit() or
  .rollback() is called.

  Transactions are created lazily.  The first operation that requires
  a transaction handle will issue the low-level BeginTransaction
  request and wait for it to return.

  Transactions keep track of the entity group.  All operations within
  a transaction must use the same entity group.  An entity group
  (currently) comprises an app id, a namespace, and a top-level key (a
  kind and an id or name).  The first operation performed determines
  the entity group.  There is some special-casing when the first
  operation is a put() of an entity with an incomplete key; in this case
  the entity group is determined after the operation returns.

  NOTE: the datastore stubs in the dev_appserver currently support
  only a single concurrent transaction.  Specifically, the (old) file
  stub locks up if an attempt is made to start a new transaction while
  a transaction is already in use, whereas the sqlite stub fails an
  assertion.
  """
    UNKNOWN_DATASTORE = 0
    PRIMARY_STANDBY_DATASTORE = 1
    HIGH_REPLICATION_DATASTORE = 2
    __SUPPORTED_VERSIONS = frozenset((_DATASTORE_V3, _CLOUD_DATASTORE_V1))

    @_positional(1)
    def __init__(self, adapter=None, config=None, _api_version=_DATASTORE_V3):
        """Constructor.

    All arguments should be specified as keyword arguments.

    Args:
      adapter: Optional AbstractAdapter subclass instance;
        default IdentityAdapter.
      config: Optional Configuration object.
    """
        if adapter is None:
            adapter = IdentityAdapter()
        if not isinstance(adapter, AbstractAdapter):
            raise datastore_errors.BadArgumentError('invalid adapter argument (%r)' % (adapter,))
        self.__adapter = adapter
        if config is None:
            config = Configuration()
        elif not Configuration.is_configuration(config):
            raise datastore_errors.BadArgumentError('invalid config argument (%r)' % (config,))
        self.__config = config
        if _api_version not in self.__SUPPORTED_VERSIONS:
            raise datastore_errors.BadArgumentError('unsupported API version (%s)' % (_api_version,))
        if _api_version == _CLOUD_DATASTORE_V1:
            if not _CLOUD_DATASTORE_ENABLED:
                raise datastore_errors.BadArgumentError(datastore_pbs.MISSING_CLOUD_DATASTORE_MESSAGE)
            apiproxy_stub_map.apiproxy.ReplaceStub(_NOOP_SERVICE, _NoopRPCStub())
        self._api_version = _api_version
        self.__pending_rpcs = set()

    @property
    def adapter(self):
        """The adapter used by this connection."""
        return self.__adapter

    @property
    def config(self):
        """The default configuration used by this connection."""
        return self.__config

    def _add_pending(self, rpc):
        """Add an RPC object to the list of pending RPCs.

    The argument must be a UserRPC object, not a MultiRpc object.
    """
        assert not isinstance(rpc, MultiRpc)
        self.__pending_rpcs.add(rpc)

    def _remove_pending(self, rpc):
        """Remove an RPC object from the list of pending RPCs.

    If the argument is a MultiRpc object, the wrapped RPCs are removed
    from the list of pending RPCs.
    """
        if isinstance(rpc, MultiRpc):
            for wrapped_rpc in rpc._MultiRpc__rpcs:
                self._remove_pending(wrapped_rpc)
        else:
            try:
                self.__pending_rpcs.remove(rpc)
            except KeyError:
                pass

    def is_pending(self, rpc):
        """Check whether an RPC object is currently pending.

    Note that 'pending' in this context refers to an RPC associated
    with this connection for which _remove_pending() hasn't been
    called yet; normally this is called by check_rpc_success() which
    itself is called by the various result hooks.  A pending RPC may
    be in the RUNNING or FINISHING state.

    If the argument is a MultiRpc object, this returns true if at least
    one of its wrapped RPCs is pending.
    """
        if isinstance(rpc, MultiRpc):
            for wrapped_rpc in rpc._MultiRpc__rpcs:
                if self.is_pending(wrapped_rpc):
                    return True
            return False
        else:
            return rpc in self.__pending_rpcs

    def get_pending_rpcs(self):
        """Return (a copy of) the list of currently pending RPCs."""
        return set(self.__pending_rpcs)

    def get_datastore_type(self, app=None):
        """Tries to get the datastore type for the given app.

    This function is only guaranteed to return something other than
    UNKNOWN_DATASTORE when running in production and querying the current app.
    """
        return _GetDatastoreType(app)

    def wait_for_all_pending_rpcs(self):
        """Wait for all currently pending RPCs to complete."""
        while self.__pending_rpcs:
            try:
                rpc = apiproxy_stub_map.UserRPC.wait_any(self.__pending_rpcs)
            except Exception:
                logging.info('wait_for_all_pending_rpcs(): exception in wait_any()', exc_info=True)
                continue
            if rpc is None:
                logging.debug('wait_any() returned None')
                continue
            assert rpc.state == apiproxy_rpc.RPC.FINISHING
            if rpc in self.__pending_rpcs:
                try:
                    self.check_rpc_success(rpc)
                except Exception:
                    logging.info('wait_for_all_pending_rpcs(): exception in check_rpc_success()', exc_info=True)

    def _create_rpc(self, config=None, service_name=None):
        """Create an RPC object using the configuration parameters.

    Internal only.

    Args:
      config: Optional Configuration object.
      service_name: Optional datastore service name.

    Returns:
      A new UserRPC object with the designated settings.

    NOTES:

    (1) The RPC object returned can only be used to make a single call
        (for details see apiproxy_stub_map.UserRPC).

    (2) To make a call, use one of the specific methods on the
        Connection object, such as conn.put(entities).  This sends the
        call to the server but does not wait.  To wait for the call to
        finish and get the result, call rpc.get_result().
    """
        deadline = Configuration.deadline(config, self.__config)
        on_completion = Configuration.on_completion(config, self.__config)
        callback = None
        if service_name is None:
            service_name = self._api_version
        if on_completion is not None:

            def callback():
                return on_completion(rpc)
        rpc = apiproxy_stub_map.UserRPC(service_name, deadline, callback)
        return rpc
    create_rpc = _create_rpc

    def _set_request_read_policy(self, request, config=None):
        """Set the read policy on a request.

    This takes the read policy from the config argument or the
    configuration's default configuration, and sets the request's read
    options.

    Args:
      request: A read request protobuf.
      config: Optional Configuration object.

    Returns:
      True if the read policy specifies a read current request, False if it
        specifies an eventually consistent request, None if it does
        not specify a read consistency.
    """
        if isinstance(config, apiproxy_stub_map.UserRPC):
            read_policy = getattr(config, 'read_policy', None)
        else:
            read_policy = Configuration.read_policy(config)
        if read_policy is None:
            read_policy = self.__config.read_policy
        if hasattr(request, 'set_failover_ms') and hasattr(request, 'strong'):
            if read_policy == Configuration.APPLY_ALL_JOBS_CONSISTENCY:
                request.set_strong(True)
                return True
            elif read_policy == Configuration.EVENTUAL_CONSISTENCY:
                request.set_strong(False)
                request.set_failover_ms(-1)
                return False
            else:
                return None
        elif hasattr(request, 'read_options'):
            if read_policy == Configuration.EVENTUAL_CONSISTENCY:
                request.read_options.read_consistency = googledatastore.ReadOptions.EVENTUAL
                return False
            else:
                return None
        else:
            raise datastore_errors.BadRequestError('read_policy is only supported on read operations.')

    def _set_request_transaction(self, request):
        """Set the current transaction on a request.

    NOTE: This version of the method does nothing.  The version
    overridden by TransactionalConnection is the real thing.

    Args:
      request: A protobuf with a transaction field.

    Returns:
      An object representing a transaction or None.
    """
        return None

    def _make_rpc_call(self, config, method, request, response, get_result_hook=None, user_data=None, service_name=None):
        """Make an RPC call.

    Internal only.

    Except for the added config argument, this is a thin wrapper
    around UserRPC.make_call().

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.
      method: The method name.
      request: The request protocol buffer.
      response: The response protocol buffer.
      get_result_hook: Optional get-result hook function.  If not None,
        this must be a function with exactly one argument, the RPC
        object (self).  Its return value is returned from get_result().
      user_data: Optional additional arbitrary data for the get-result
        hook function.  This can be accessed as rpc.user_data.  The
        type of this value is up to the service module.

    Returns:
      The UserRPC object used for the call.
    """
        if isinstance(config, apiproxy_stub_map.UserRPC):
            rpc = config
        else:
            rpc = self._create_rpc(config, service_name)
        rpc.make_call(six_subset.ensure_binary(method), request, response, get_result_hook, user_data)
        self._add_pending(rpc)
        return rpc
    make_rpc_call = _make_rpc_call

    def check_rpc_success(self, rpc):
        """Check for RPC success and translate exceptions.

    This wraps rpc.check_success() and should be called instead of that.

    This also removes the RPC from the list of pending RPCs, once it
    has completed.

    Args:
      rpc: A UserRPC or MultiRpc object.

    Raises:
      Nothing if the call succeeded; various datastore_errors.Error
      subclasses if ApplicationError was raised by rpc.check_success().
    """
        try:
            rpc.wait()
        finally:
            self._remove_pending(rpc)
        try:
            rpc.check_success()
        except apiproxy_errors.ApplicationError as err:
            raise _ToDatastoreError(err)
    MAX_RPC_BYTES = 1024 * 1024
    MAX_GET_KEYS = 1000
    MAX_PUT_ENTITIES = 500
    MAX_DELETE_KEYS = 500
    MAX_ALLOCATE_IDS_KEYS = 500
    DEFAULT_MAX_ENTITY_GROUPS_PER_RPC = 10

    def __get_max_entity_groups_per_rpc(self, config):
        """Internal helper: figures out max_entity_groups_per_rpc for the config."""
        return Configuration.max_entity_groups_per_rpc(config, self.__config) or self.DEFAULT_MAX_ENTITY_GROUPS_PER_RPC

    def _extract_entity_group(self, value):
        """Internal helper: extracts the entity group from a key or entity.

    Supports both v3 and v1 protobufs.

    Args:
      value: an entity_pb.{Reference, EntityProto} or
          googledatastore.{Key, Entity}.

    Returns:
      A tuple consisting of:
        - kind
        - name, id, or ('new', unique id)
    """
        if _CLOUD_DATASTORE_ENABLED and isinstance(value, googledatastore.Entity):
            value = value.key
        if isinstance(value, entity_pb.EntityProto):
            value = value.key()
        if _CLOUD_DATASTORE_ENABLED and isinstance(value, googledatastore.Key):
            elem = value.path[0]
            elem_id = elem.id
            elem_name = elem.name
            kind = elem.kind
        else:
            elem = value.path().element(0)
            kind = elem.type()
            elem_id = elem.id()
            elem_name = elem.name()
        return (kind, elem_id or elem_name or ('new', id(elem)))

    def _map_and_group(self, values, map_fn, group_fn):
        """Internal helper: map values to keys and group by key. Here key is any
    object derived from an input value by map_fn, and which can be grouped
    by group_fn.

    Args:
      values: The values to be grouped by applying get_group(to_ref(value)).
      map_fn: a function that maps a value to a key to be grouped.
      group_fn: a function that groups the keys output by map_fn.

    Returns:
      A list where each element is a list of (key, index) pairs.  Here
      index is the location of the value from which the key was derived in
      the original list.
    """
        indexed_key_groups = collections.defaultdict(list)
        for index, value in enumerate(values):
            key = map_fn(value)
            indexed_key_groups[group_fn(key)].append((key, index))
        return list(indexed_key_groups.values())

    def __create_result_index_pairs(self, indexes):
        """Internal helper: build a function that ties an index with each result.

    Args:
      indexes: A list of integers.  A value x at location y in the list means
        that the result at location y in the result list needs to be at location
        x in the list of results returned to the user.
    """

        def create_result_index_pairs(results):
            return list(zip(results, indexes))
        return create_result_index_pairs

    def __sort_result_index_pairs(self, extra_hook):
        """Builds a function that sorts the indexed results.

    Args:
      extra_hook: A function that the returned function will apply to its result
        before returning.

    Returns:
      A function that takes a list of results and reorders them to match the
      order in which the input values associated with each results were
      originally provided.
    """

        def sort_result_index_pairs(result_index_pairs):
            results = [None] * len(result_index_pairs)
            for result, index in result_index_pairs:
                results[index] = result
            if extra_hook is not None:
                results = extra_hook(results)
            return results
        return sort_result_index_pairs

    def _generate_pb_lists(self, grouped_values, base_size, max_count, max_groups, config):
        """Internal helper: repeatedly yield a list of 2 elements.

    Args:
      grouped_values: A list of lists.  The inner lists consist of objects
        grouped by e.g. entity group or id sequence.

      base_size: An integer representing the base size of an rpc.  Used for
        splitting operations across multiple RPCs due to size limitations.

      max_count: An integer representing the maximum number of objects we can
        send in an rpc.  Used for splitting operations across multiple RPCs.

      max_groups: An integer representing the maximum number of groups we can
        have represented in an rpc.  Can be None, in which case no constraint.

      config: The config object, defining max rpc size in bytes.

    Yields:
      Repeatedly yields 2 element tuples.  The first element is a list of
      protobufs to send in one batch.  The second element is a list containing
      the original location of those protobufs (expressed as an index) in the
      input.
    """
        max_size = Configuration.max_rpc_bytes(config, self.__config) or self.MAX_RPC_BYTES
        pbs = []
        pb_indexes = []
        size = base_size
        num_groups = 0
        for indexed_pbs in grouped_values:
            num_groups += 1
            if max_groups is not None and num_groups > max_groups:
                yield (pbs, pb_indexes)
                pbs = []
                pb_indexes = []
                size = base_size
                num_groups = 1
            for indexed_pb in indexed_pbs:
                pb, index = indexed_pb
                incr_size = pb.ByteSize() + 5
                if not isinstance(config, apiproxy_stub_map.UserRPC) and (len(pbs) >= max_count or (pbs and size + incr_size > max_size)):
                    yield (pbs, pb_indexes)
                    pbs = []
                    pb_indexes = []
                    size = base_size
                    num_groups = 1
                pbs.append(pb)
                pb_indexes.append(index)
                size += incr_size
        yield (pbs, pb_indexes)

    def __force(self, req):
        """Configure a request to force mutations."""
        if isinstance(req, (datastore_pb.PutRequest, datastore_pb.TouchRequest, datastore_pb.DeleteRequest)):
            req.set_force(True)

    def get(self, keys):
        """Synchronous Get operation.

    Args:
      keys: An iterable of user-level key objects.

    Returns:
      A list of user-level entity objects and None values, corresponding
      1:1 to the argument keys.  A None means there is no entity for the
      corresponding key.
    """
        return self.async_get(None, keys).get_result()

    def async_get(self, config, keys, extra_hook=None):
        """Asynchronous Get operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.
      keys: An iterable of user-level key objects.
      extra_hook: Optional function to be called on the result once the
        RPC has completed.

    Returns:
      A MultiRpc object.
    """

        def make_get_call(base_req, pbs, extra_hook=None):
            req = copy.deepcopy(base_req)
            if self._api_version == _CLOUD_DATASTORE_V1:
                method = 'Lookup'
                req.keys.extend(pbs)
                resp = googledatastore.LookupResponse()
            else:
                method = 'Get'
                req.key_list().extend(pbs)
                resp = datastore_pb.GetResponse()
            user_data = (config, pbs, extra_hook)
            return self._make_rpc_call(config, method, req, resp, get_result_hook=self.__get_hook, user_data=user_data, service_name=self._api_version)
        if self._api_version == _CLOUD_DATASTORE_V1:
            base_req = googledatastore.LookupRequest()
            key_to_pb = self.__adapter.key_to_pb_v1
        else:
            base_req = datastore_pb.GetRequest()
            base_req.set_allow_deferred(True)
            key_to_pb = self.__adapter.key_to_pb
        is_read_current = self._set_request_read_policy(base_req, config)
        txn = self._set_request_transaction(base_req)
        if isinstance(config, apiproxy_stub_map.UserRPC) or len(keys) <= 1:
            pbs = [key_to_pb(key) for key in keys]
            return make_get_call(base_req, pbs, extra_hook)
        max_count = Configuration.max_get_keys(config, self.__config) or self.MAX_GET_KEYS
        indexed_keys_by_entity_group = self._map_and_group(keys, key_to_pb, self._extract_entity_group)
        if is_read_current is None:
            is_read_current = self.get_datastore_type() == BaseConnection.HIGH_REPLICATION_DATASTORE
        if is_read_current and txn is None:
            max_egs_per_rpc = self.__get_max_entity_groups_per_rpc(config)
        else:
            max_egs_per_rpc = None
        pbsgen = self._generate_pb_lists(indexed_keys_by_entity_group, base_req.ByteSize(), max_count, max_egs_per_rpc, config)
        rpcs = []
        for pbs, indexes in pbsgen:
            rpcs.append(make_get_call(base_req, pbs, self.__create_result_index_pairs(indexes)))
        return MultiRpc(rpcs, self.__sort_result_index_pairs(extra_hook))

    def __get_hook(self, rpc):
        """Internal method used as get_result_hook for Get operation."""
        self.check_rpc_success(rpc)
        config, keys_from_request, extra_hook = rpc.user_data
        if self._api_version == _DATASTORE_V3 and rpc.response.in_order():
            entities = []
            for entity_result in rpc.response.entity_list():
                if entity_result.has_entity():
                    entity = self.__adapter.pb_to_entity(entity_result.entity())
                else:
                    entity = None
                entities.append(entity)
        else:
            current_get_response = rpc.response
            result_dict = {}
            self.__add_get_response_entities_to_dict(current_get_response, result_dict)
            deferred_req = copy.deepcopy(rpc.request)
            if self._api_version == _CLOUD_DATASTORE_V1:
                method = 'Lookup'
                deferred_resp = googledatastore.LookupResponse()
                while current_get_response.deferred:
                    deferred_req.ClearField('keys')
                    deferred_req.keys.extend(current_get_response.deferred)
                    deferred_resp.Clear()
                    deferred_rpc = self._make_rpc_call(config, method, deferred_req, deferred_resp, service_name=self._api_version)
                    deferred_rpc.get_result()
                    current_get_response = deferred_rpc.response
                    self.__add_get_response_entities_to_dict(current_get_response, result_dict)
            else:
                method = 'Get'
                deferred_resp = datastore_pb.GetResponse()
                while current_get_response.deferred_list():
                    deferred_req.clear_key()
                    deferred_req.key_list().extend(current_get_response.deferred_list())
                    deferred_resp.Clear()
                    deferred_rpc = self._make_rpc_call(config, method, deferred_req, deferred_resp, service_name=self._api_version)
                    deferred_rpc.get_result()
                    current_get_response = deferred_rpc.response
                    self.__add_get_response_entities_to_dict(current_get_response, result_dict)
            entities = [result_dict.get(datastore_types.ReferenceToKeyValue(pb)) for pb in keys_from_request]
        if extra_hook is not None:
            entities = extra_hook(entities)
        return entities

    def __add_get_response_entities_to_dict(self, get_response, result_dict):
        """Converts entities from the get response and adds them to the dict.

    The Key for the dict will be calculated via
    datastore_types.ReferenceToKeyValue.  There will be no entry for entities
    that were not found.

    Args:
      get_response: A datastore_pb.GetResponse or
          googledatastore.LookupResponse.
      result_dict: The dict to add results to.
    """
        if _CLOUD_DATASTORE_ENABLED and isinstance(get_response, googledatastore.LookupResponse):
            for result in get_response.found:
                v1_key = result.entity.key
                entity = self.__adapter.pb_v1_to_entity(result.entity, False)
                result_dict[datastore_types.ReferenceToKeyValue(v1_key)] = entity
        else:
            for entity_result in get_response.entity_list():
                if entity_result.has_entity():
                    reference_pb = entity_result.entity().key()
                    hashable_key = datastore_types.ReferenceToKeyValue(reference_pb)
                    entity = self.__adapter.pb_to_entity(entity_result.entity())
                    result_dict[hashable_key] = entity

    def get_indexes(self):
        """Synchronous get indexes operation.

    Returns:
      user-level indexes representation
    """
        return self.async_get_indexes(None).get_result()

    def async_get_indexes(self, config, extra_hook=None, _app=None):
        """Asynchronous get indexes operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.
      extra_hook: Optional function to be called once the RPC has completed.

    Returns:
      A MultiRpc object.
    """
        req = datastore_pb.GetIndicesRequest()
        req.set_app_id(datastore_types.ResolveAppId(_app))
        resp = datastore_pb.CompositeIndices()
        return self._make_rpc_call(config, 'GetIndices', req, resp, get_result_hook=self.__get_indexes_hook, user_data=extra_hook, service_name=_DATASTORE_V3)

    def __get_indexes_hook(self, rpc):
        """Internal method used as get_result_hook for Get operation."""
        self.check_rpc_success(rpc)
        indexes = [self.__adapter.pb_to_index(index) for index in rpc.response.index_list()]
        if rpc.user_data:
            indexes = rpc.user_data(indexes)
        return indexes

    def put(self, entities):
        """Synchronous Put operation.

    Args:
      entities: An iterable of user-level entity objects.

    Returns:
      A list of user-level key objects, corresponding 1:1 to the
      argument entities.

    NOTE: If any of the entities has an incomplete key, this will
    *not* patch up those entities with the complete key.
    """
        return self.async_put(None, entities).get_result()

    def async_put(self, config, entities, extra_hook=None):
        """Asynchronous Put operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.
      entities: An iterable of user-level entity objects.
      extra_hook: Optional function to be called on the result once the
        RPC has completed.

     Returns:
      A MultiRpc object.

    NOTE: If any of the entities has an incomplete key, this will
    *not* patch up those entities with the complete key.
    """

        def make_put_call(base_req, pbs, user_data=None):
            req = copy.deepcopy(base_req)
            if self._api_version == _CLOUD_DATASTORE_V1:
                for entity in pbs:
                    mutation = req.mutations.add()
                    mutation.upsert.CopyFrom(entity)
                method = 'Commit'
                resp = googledatastore.CommitResponse()
            else:
                req.entity_list().extend(pbs)
                method = 'Put'
                resp = datastore_pb.PutResponse()
            user_data = (pbs, user_data)
            return self._make_rpc_call(config, method, req, resp, get_result_hook=self.__put_hook, user_data=user_data, service_name=self._api_version)
        if self._api_version == _CLOUD_DATASTORE_V1:
            base_req = googledatastore.CommitRequest()
            base_req.mode = googledatastore.CommitRequest.NON_TRANSACTIONAL
            entity_to_pb = self.__adapter.entity_to_pb_v1
        else:
            base_req = datastore_pb.PutRequest()
            entity_to_pb = self.__adapter.entity_to_pb
        self._set_request_transaction(base_req)
        if Configuration.force_writes(config, self.__config):
            self.__force(base_req)
        if isinstance(config, apiproxy_stub_map.UserRPC) or len(entities) <= 1:
            pbs = [entity_to_pb(entity) for entity in entities]
            return make_put_call(base_req, pbs, extra_hook)
        max_count = Configuration.max_put_entities(config, self.__config) or self.MAX_PUT_ENTITIES
        if self._api_version == _CLOUD_DATASTORE_V1 and (not base_req.transaction) or not base_req.has_transaction():
            max_egs_per_rpc = self.__get_max_entity_groups_per_rpc(config)
        else:
            max_egs_per_rpc = None
        indexed_entities_by_entity_group = self._map_and_group(entities, entity_to_pb, self._extract_entity_group)
        pbsgen = self._generate_pb_lists(indexed_entities_by_entity_group, base_req.ByteSize(), max_count, max_egs_per_rpc, config)
        rpcs = []
        for pbs, indexes in pbsgen:
            rpcs.append(make_put_call(base_req, pbs, self.__create_result_index_pairs(indexes)))
        return MultiRpc(rpcs, self.__sort_result_index_pairs(extra_hook))

    def __put_hook(self, rpc):
        """Internal method used as get_result_hook for Put operation."""
        self.check_rpc_success(rpc)
        entities_from_request, extra_hook = rpc.user_data
        if _CLOUD_DATASTORE_ENABLED and isinstance(rpc.response, googledatastore.CommitResponse):
            keys = []
            i = 0
            for entity in entities_from_request:
                if datastore_pbs.is_complete_v1_key(entity.key):
                    keys.append(entity.key)
                else:
                    keys.append(rpc.response.mutation_results[i].key)
                    i += 1
            keys = [self.__adapter.pb_v1_to_key(key) for key in keys]
        else:
            keys = [self.__adapter.pb_to_key(key) for key in rpc.response.key_list()]
        if extra_hook is not None:
            keys = extra_hook(keys)
        return keys

    def delete(self, keys):
        """Synchronous Delete operation.

    Args:
      keys: An iterable of user-level key objects.

    Returns:
      None.
    """
        return self.async_delete(None, keys).get_result()

    def async_delete(self, config, keys, extra_hook=None):
        """Asynchronous Delete operation.

    Args:
      config: A Configuration object or None.  Defaults are taken from
        the connection's default configuration.
      keys: An iterable of user-level key objects.
      extra_hook: Optional function to be called once the RPC has completed.

    Returns:
      A MultiRpc object.
    """

        def make_delete_call(base_req, pbs, user_data=None):
            req = copy.deepcopy(base_req)
            if self._api_version == _CLOUD_DATASTORE_V1:
                for pb in pbs:
                    mutation = req.mutations.add()
                    mutation.delete.CopyFrom(pb)
                method = 'Commit'
                resp = googledatastore.CommitResponse()
            else:
                req.key_list().extend(pbs)
                method = 'Delete'
                resp = datastore_pb.DeleteResponse()
            return self._make_rpc_call(config, method, req, resp, get_result_hook=self.__delete_hook, user_data=user_data, service_name=self._api_version)
        if self._api_version == _CLOUD_DATASTORE_V1:
            base_req = googledatastore.CommitRequest()
            base_req.mode = googledatastore.CommitRequest.NON_TRANSACTIONAL
            key_to_pb = self.__adapter.key_to_pb_v1
        else:
            base_req = datastore_pb.DeleteRequest()
            key_to_pb = self.__adapter.key_to_pb
        self._set_request_transaction(base_req)
        if Configuration.force_writes(config, self.__config):
            self.__force(base_req)
        if isinstance(config, apiproxy_stub_map.UserRPC) or len(keys) <= 1:
            pbs = [key_to_pb(key) for key in keys]
            return make_delete_call(base_req, pbs, extra_hook)
        max_count = Configuration.max_delete_keys(config, self.__config) or self.MAX_DELETE_KEYS
        if self._api_version == _CLOUD_DATASTORE_V1 and (not base_req.transaction) or not base_req.has_transaction():
            max_egs_per_rpc = self.__get_max_entity_groups_per_rpc(config)
        else:
            max_egs_per_rpc = None
        indexed_keys_by_entity_group = self._map_and_group(keys, key_to_pb, self._extract_entity_group)
        pbsgen = self._generate_pb_lists(indexed_keys_by_entity_group, base_req.ByteSize(), max_count, max_egs_per_rpc, config)
        rpcs = []
        for pbs, _ in pbsgen:
            rpcs.append(make_delete_call(base_req, pbs))
        return MultiRpc(rpcs, extra_hook)

    def __delete_hook(self, rpc):
        """Internal method used as get_result_hook for Delete operation."""
        self.check_rpc_success(rpc)
        if rpc.user_data is not None:
            rpc.user_data(None)

    def begin_transaction(self, app, previous_transaction=None, mode=TransactionMode.UNKNOWN):
        """Synchronous BeginTransaction operation.

    NOTE: In most cases the new_transaction() method is preferred,
    since that returns a TransactionalConnection object which will
    begin the transaction lazily.

    Args:
      app: Application ID.
      previous_transaction: The transaction to reset.
      mode: The transaction mode.

    Returns:
      An object representing a transaction or None.
    """
        return self.async_begin_transaction(None, app, previous_transaction, mode).get_result()

    def async_begin_transaction(self, config, app, previous_transaction=None, mode=TransactionMode.UNKNOWN):
        """Asynchronous BeginTransaction operation.

    Args:
      config: A configuration object or None.  Defaults are taken from
        the connection's default configuration.
      app: Application ID.
      previous_transaction: The transaction to reset.
      mode: The transaction mode.

    Returns:
      A MultiRpc object.
    """
        if not isinstance(app, six_subset.string_types) or not app:
            raise datastore_errors.BadArgumentError('begin_transaction requires an application id argument (%r)' % (app,))
        if previous_transaction is not None and mode == TransactionMode.READ_ONLY:
            raise datastore_errors.BadArgumentError('begin_transaction requires mode != READ_ONLY when previous_transaction is not None')
        if self._api_version == _CLOUD_DATASTORE_V1:
            req = googledatastore.BeginTransactionRequest()
            resp = googledatastore.BeginTransactionResponse()
            if previous_transaction is not None:
                mode = TransactionMode.READ_WRITE
            if mode == TransactionMode.UNKNOWN:
                pass
            elif mode == TransactionMode.READ_ONLY:
                req.transaction_options.read_only.SetInParent()
            elif mode == TransactionMode.READ_WRITE:
                if previous_transaction is not None:
                    req.transaction_options.read_write.previous_transaction = previous_transaction
                else:
                    req.transaction_options.read_write.SetInParent()
        else:
            req = datastore_pb.BeginTransactionRequest()
            req.set_app(app)
            if TransactionOptions.xg(config, self.__config):
                req.set_allow_multiple_eg(True)
            if mode == TransactionMode.UNKNOWN:
                pass
            elif mode == TransactionMode.READ_ONLY:
                req.set_mode(datastore_pb.BeginTransactionRequest.READ_ONLY)
            elif mode == TransactionMode.READ_WRITE:
                req.set_mode(datastore_pb.BeginTransactionRequest.READ_WRITE)
            if previous_transaction is not None:
                req.mutable_previous_transaction().CopyFrom(previous_transaction)
            resp = datastore_pb.Transaction()
        return self._make_rpc_call(config, 'BeginTransaction', req, resp, get_result_hook=self.__begin_transaction_hook, service_name=self._api_version)

    def __begin_transaction_hook(self, rpc):
        """Internal method used as get_result_hook for BeginTransaction."""
        self.check_rpc_success(rpc)
        if self._api_version == _CLOUD_DATASTORE_V1:
            return rpc.response.transaction
        else:
            return rpc.response