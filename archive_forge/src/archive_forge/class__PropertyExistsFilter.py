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
class _PropertyExistsFilter(FilterPredicate):
    """A FilterPredicate that matches entities containing specific properties.

  Only works as an in-memory filter. Used internally to filter out entities
  that don't have all properties in a given Order.
  """

    def __init__(self, names):
        super(_PropertyExistsFilter, self).__init__()
        self._names = frozenset(names)

    def _apply(self, value_map):
        for name in self._names:
            if not value_map.get(name):
                return False
        return True

    def _get_prop_names(self):
        return self._names

    def _prune(self, _):
        raise NotImplementedError

    def __getstate__(self):
        raise pickle.PicklingError('Pickling of %r is unsupported.' % self)