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
class FetchOptions(datastore_rpc.Configuration):
    """An immutable class that contains all options for fetching results.

  These options apply to any request that pulls results from a query.

  This class reserves the right to define configuration options of any name
  except those that start with 'user_'. External subclasses should only define
  function or variables with names that start with in 'user_'.

  Options are set by passing keyword arguments to the constructor corresponding
  to the configuration options defined below and in datastore_rpc.Configuration.

  This object can be used as the default config for a datastore_rpc.Connection
  but in that case some options will be ignored, see option documentation below
  for details.
  """

    @datastore_rpc.ConfigOption
    def produce_cursors(value):
        """If a Cursor should be returned with the fetched results.

    Raises:
      datastore_errors.BadArgumentError if value is not a bool.
    """
        if not isinstance(value, bool):
            raise datastore_errors.BadArgumentError('produce_cursors argument should be bool (%r)' % (value,))
        return value

    @datastore_rpc.ConfigOption
    def offset(value):
        """The number of results to skip before returning the first result.

    Only applies to the first request it is used with and is ignored if present
    on datastore_rpc.Connection.config.

    Raises:
      datastore_errors.BadArgumentError if value is not a integer or is less
      than zero.
    """
        datastore_types.ValidateInteger(value, 'offset', datastore_errors.BadArgumentError, zero_ok=True)
        return value

    @datastore_rpc.ConfigOption
    def batch_size(value):
        """The number of results to attempt to retrieve in a batch.

    Raises:
      datastore_errors.BadArgumentError if value is not a integer or is not
      greater than zero.
    """
        datastore_types.ValidateInteger(value, 'batch_size', datastore_errors.BadArgumentError)
        return value