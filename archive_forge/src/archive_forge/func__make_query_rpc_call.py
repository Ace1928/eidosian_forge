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
def _make_query_rpc_call(self, config, req):
    """Makes a RunQuery call that will modify the instance.

    Args:
      config: The datastore_rpc.Configuration to use for the call.
      req: The request to send with the call.

    Returns:
      A UserRPC object that can be used to fetch the result of the RPC.
    """
    _api_version = self._batch_shared.conn._api_version
    if _api_version == datastore_rpc._CLOUD_DATASTORE_V1:
        return self._batch_shared.conn._make_rpc_call(config, 'RunQuery', req, googledatastore.RunQueryResponse(), self.__v1_run_query_response_hook)
    return self._batch_shared.conn._make_rpc_call(config, 'RunQuery', req, datastore_pb.QueryResult(), self.__query_result_hook)