from __future__ import absolute_import
from __future__ import unicode_literals
import base64
import collections
import pickle
from googlecloudsdk.third_party.appengine.proto import ProtocolBuffer
from googlecloudsdk.third_party.appengine.googlestorage.onestore.v3 import entity_pb
from googlecloudsdk.third_party.appengine._internal import six_subset
from googlecloudsdk.third_party.appengine.api import datastore_errors
from googlecloudsdk.third_party.appengine.api import datastore_types
from googlecloudsdk.third_party.appengine.datastore import datastore_index
from googlecloudsdk.third_party.appengine.datastore import datastore_pb
from googlecloudsdk.third_party.appengine.datastore import datastore_pbs
from googlecloudsdk.third_party.appengine.datastore import datastore_rpc
class _QueryKeyFilter(_BaseComponent):
    """A class that implements the key filters available on a Query."""

    @datastore_rpc._positional(1)
    def __init__(self, app=None, namespace=None, kind=None, ancestor=None):
        """Constructs a _QueryKeyFilter.

    If app/namespace and ancestor are not defined, the app/namespace set in the
    environment is used.

    Args:
      app: a string representing the required app id or None.
      namespace: a string representing the required namespace or None.
      kind: a string representing the required kind or None.
      ancestor: a entity_pb.Reference representing the required ancestor or
        None.

    Raises:
      datastore_erros.BadArgumentError if app and ancestor.app() do not match or
        an unexpected type is passed in for any argument.
    """
        if kind is not None:
            datastore_types.ValidateString(kind, 'kind', datastore_errors.BadArgumentError)
        if ancestor is not None:
            if not isinstance(ancestor, entity_pb.Reference):
                raise datastore_errors.BadArgumentError('ancestor argument should be entity_pb.Reference (%r)' % (ancestor,))
            if app is None:
                app = ancestor.app()
            elif app != ancestor.app():
                raise datastore_errors.BadArgumentError('ancestor argument should match app ("%r" != "%r")' % (ancestor.app(), app))
            if namespace is None:
                namespace = ancestor.name_space()
            elif namespace != ancestor.name_space():
                raise datastore_errors.BadArgumentError('ancestor argument should match namespace ("%r" != "%r")' % (ancestor.name_space(), namespace))
            pb = entity_pb.Reference()
            pb.CopyFrom(ancestor)
            ancestor = pb
            self.__ancestor = ancestor
            self.__path = ancestor.path().element_list()
        else:
            self.__ancestor = None
            self.__path = None
        super(_QueryKeyFilter, self).__init__()
        self.__app = datastore_types.ResolveAppId(app).encode('utf-8')
        self.__namespace = datastore_types.ResolveNamespace(namespace).encode('utf-8')
        self.__kind = kind and kind.encode('utf-8')

    @property
    def app(self):
        return self.__app

    @property
    def namespace(self):
        return self.__namespace

    @property
    def kind(self):
        return self.__kind

    @property
    def ancestor(self):
        return self.__ancestor

    def __call__(self, entity_or_reference):
        """Apply the filter.

    Accepts either an entity or a reference to avoid the need to extract keys
    from entities when we have a list of entities (which is a common case).

    Args:
      entity_or_reference: Either an entity_pb.EntityProto or
        entity_pb.Reference.
    """
        if isinstance(entity_or_reference, entity_pb.Reference):
            key = entity_or_reference
        elif isinstance(entity_or_reference, entity_pb.EntityProto):
            key = entity_or_reference.key()
        else:
            raise datastore_errors.BadArgumentError('entity_or_reference argument must be an entity_pb.EntityProto ' + 'or entity_pb.Reference (%r)' % entity_or_reference)
        return key.app() == self.__app and key.name_space() == self.__namespace and (not self.__kind or key.path().element_list()[-1].type() == self.__kind) and (not self.__path or key.path().element_list()[0:len(self.__path)] == self.__path)

    def _to_pb(self):
        """Returns an internal pb representation."""
        pb = datastore_pb.Query()
        pb.set_app(self.__app)
        datastore_types.SetNamespace(pb, self.__namespace)
        if self.__kind is not None:
            pb.set_kind(self.__kind)
        if self.__ancestor:
            ancestor = pb.mutable_ancestor()
            ancestor.CopyFrom(self.__ancestor)
        return pb

    def _to_pb_v1(self, adapter):
        """Returns a v1 internal proto representation of the query key filter.

    Args:
      adapter: A datastore_rpc.AbstractAdapter.
    Returns:
      A tuple (googledatastore.RunQueryRequest, googledatastore.Filter).

    The second tuple value is a Filter representing the ancestor portion of the
    query. If there is no ancestor constraint, this value will be None
    """
        pb = googledatastore.RunQueryRequest()
        partition_id = pb.partition_id
        partition_id.project_id = adapter.get_entity_converter().app_to_project_id(self.__app)
        if self.__namespace:
            partition_id.namespace_id = self.__namespace
        if self.__kind is not None:
            pb.query.kind.add().name = self.__kind
        ancestor_filter = None
        if self.__ancestor:
            ancestor_filter = googledatastore.Filter()
            ancestor_prop_filter = ancestor_filter.property_filter
            ancestor_prop_filter.op = googledatastore.PropertyFilter.HAS_ANCESTOR
            prop_pb = ancestor_prop_filter.property
            prop_pb.name = datastore_types.KEY_SPECIAL_PROPERTY
            adapter.get_entity_converter().v3_to_v1_key(self.ancestor, ancestor_prop_filter.value.key_value)
        return (pb, ancestor_filter)