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
class FilterPredicate(_PropertyComponent):
    """An abstract base class for all query filters.

  All sub-classes must be immutable as these are often stored without creating a
  defensive copy.
  """

    def __call__(self, entity):
        """Applies the filter predicate to the given entity.

    Args:
      entity: the datastore_pb.EntityProto to test.

    Returns:
      True if the given entity matches the filter, False otherwise.
    """
        return self._apply(_make_key_value_map(entity, self._get_prop_names()))

    def _apply(self, key_value_map):
        """Apply the given component to the comparable value map.

    A filter matches a list of values if at least one value in the list
    matches the filter, for example:
      'prop: [1, 2]' matches both 'prop = 1' and 'prop = 2' but not 'prop = 3'

    Note: the values are actually represented as tuples whose first item
    encodes the type; see datastore_types.PropertyValueToKeyValue().

    Args:
      key_value_map: A dict mapping property names to a list of
        comparable values.

    Return:
      A boolean indicating if the given map matches the filter.
    """
        raise NotImplementedError

    def _prune(self, key_value_map):
        """Removes values from the given map that do not match the filter.

    When doing a scan in the datastore, only index values that match the filters
    are seen. When multiple values that point to the same entity are seen, the
    entity only appears where the first value is found. This function removes
    all values that don't match the query so that the first value in the map
    is the same one the datastore would see first.

    Args:
      key_value_map: the comparable value map from which to remove
        values. Does not need to contain values for all filtered properties.

    Returns:
      A value that evaluates to False if every value in a single list was
      completely removed. This effectively applies the filter but is less
      efficient than _apply().
    """
        raise NotImplementedError

    def _to_pb(self):
        """Internal only function to generate a pb."""
        raise NotImplementedError('This filter only supports in memory operations (%r)' % self)

    def _to_pbs(self):
        """Internal only function to generate a list of pbs."""
        return [self._to_pb()]

    def _to_pb_v1(self, adapter):
        """Internal only function to generate a v1 pb.

    Args:
      adapter: A datastore_rpc.AbstractAdapter
    """
        raise NotImplementedError('This filter only supports in memory operations (%r)' % self)