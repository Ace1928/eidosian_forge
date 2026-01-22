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
class CorrelationFilter(FilterPredicate):
    """A filter that isolates correlated values and applies a sub-filter on them.

  This filter assumes that every property used by the sub-filter should be
  grouped before being passed to the sub-filter. The default grouping puts
  each value in its own group. Consider:
    e = {a: [1, 2], b: [2, 1, 3], c: 4}

  A correlation filter with a sub-filter that operates on (a, b) will be tested
  against the following 3 sets of values:
    {a: 1, b: 2}
    {a: 2, b: 1}
    {b: 3}

  In this case CorrelationFilter('a = 2 AND b = 2') won't match this entity but
  CorrelationFilter('a = 2 AND b = 1') will. To apply an uncorrelated filter on
  c, the filter must be applied in parallel to the correlation filter. For
  example:
    CompositeFilter(AND, [CorrelationFilter('a = 2 AND b = 1'), 'c = 3'])

  If 'c = 3' was included in the correlation filter, c would be grouped as well.
  This would result in the following values:
    {a: 1, b: 2, c: 3}
    {a: 2, b: 1}
    {b: 3}

  If any set of correlated values match the sub-filter then the entity matches
  the correlation filter.
  """

    def __init__(self, subfilter):
        """Constructor.

    Args:
      subfilter: A FilterPredicate to apply to the correlated values
    """
        self._subfilter = subfilter

    @property
    def subfilter(self):
        return self._subfilter

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self.subfilter)

    def _apply(self, value_map):
        base_map = dict(((prop, []) for prop in self._get_prop_names()))
        value_maps = []
        for prop in base_map:
            grouped = self._group_values(prop, value_map[prop])
            while len(value_maps) < len(grouped):
                value_maps.append(base_map.copy())
            for value, m in zip(grouped, value_maps):
                m[prop] = value
        return self._apply_correlated(value_maps)

    def _apply_correlated(self, value_maps):
        """Applies sub-filter to the correlated value maps.

    The default implementation matches when any value_map in value_maps
    matches the sub-filter.

    Args:
      value_maps: A list of correlated value_maps.
    Returns:
      True if any the entity matches the correlation filter.
    """
        for map in value_maps:
            if self._subfilter._apply(map):
                return True
        return False

    def _group_values(self, prop, values):
        """A function that groups the given values.

    Override this function to introduce custom grouping logic. The default
    implementation assumes each value belongs in its own group.

    Args:
      prop: The name of the property who's values are being grouped.
      values: A list of opaque values.

   Returns:
      A list of lists of grouped values.
    """
        return [[value] for value in values]

    def _get_prop_names(self):
        return self._subfilter._get_prop_names()