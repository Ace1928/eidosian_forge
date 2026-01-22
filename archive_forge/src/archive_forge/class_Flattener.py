from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.core import log
from googlecloudsdk.core.resource import resource_filter
from googlecloudsdk.core.resource import resource_printer_base
from googlecloudsdk.core.resource import resource_projector
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import peek_iterable
class Flattener(peek_iterable.Tap):
    """A Tapper class that flattens a resource key slice to separate records.

  A serialized copy of the resource is modified in place. This means the same
  resource object is returned for each flattened slice item. This is OK because
  the downstream is not guaranteed uniqueness.

  Attributes:
    _child_name: The flattened value to set is _parent_key[_child_name].
    _key: The parsed resource key of the slice to flatten.
    _parent_key: The parent of _key, None for the resource itself.
    _items: The items to flatten in the current resource.
    _resource: The serialized copy of the current resource.
  """

    def __init__(self, key):
        """Constructor.

    Args:
      key: The resource key of the slice to flatten.
    """
        self._key = key[:]
        self._child_name = self._key[-1] if self._key else None
        self._parent_key = self._key[:-1] if self._key else None
        self._items = None
        self._resource = None

    def Tap(self, resource):
        """Returns the next slice item in resource.

    Args:
      resource: The resource to flatten.

    Returns:
      True if the next slice is not a list, False if there are no more items,
      or Injector(resource) which is the resource with the next slice flattened.
    """
        if self._items is None:
            self._resource = resource_projector.MakeSerializable(resource)
            self._items = resource_property.Get(self._resource, self._key)
            if not isinstance(self._items, list):
                item = self._items
                self._items = None
                return peek_iterable.TapInjector(item, replace=True)
        if not self._items:
            self._items = None
            return False
        item = self._items.pop(0)
        if self._parent_key:
            parent = resource_property.Get(self._resource, self._parent_key)
        else:
            parent = self._resource
        parent[self._child_name] = item
        return peek_iterable.TapInjector(resource_projector.MakeSerializable(self._resource))