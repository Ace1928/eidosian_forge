import abc
import collections
import inspect
import sys
import uuid
import random
from .._utils import patch_collections_abc, stringify_id, OrderedSet
def _set_random_id(self):
    if hasattr(self, 'id'):
        return getattr(self, 'id')
    kind = f'`{self._namespace}.{self._type}`'
    if getattr(self, 'persistence', False):
        raise RuntimeError(f'\n                Attempting to use an auto-generated ID with the `persistence` prop.\n                This is prohibited because persistence is tied to component IDs and\n                auto-generated IDs can easily change.\n\n                Please assign an explicit ID to this {kind} component.\n                ')
    if 'dash_snapshots' in sys.modules:
        raise RuntimeError(f'\n                Attempting to use an auto-generated ID in an app with `dash_snapshots`.\n                This is prohibited because snapshots saves the whole app layout,\n                including component IDs, and auto-generated IDs can easily change.\n                Callbacks referencing the new IDs will not work with old snapshots.\n\n                Please assign an explicit ID to this {kind} component.\n                ')
    v = str(uuid.UUID(int=rd.randint(0, 2 ** 128)))
    setattr(self, 'id', v)
    return v