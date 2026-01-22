from . import errors
from . import graph as _mod_graph
from . import osutils, ui
def _update_needed_children(self, key, parent_keys):
    for parent_key in parent_keys:
        if parent_key in self._num_needed_children:
            self._num_needed_children[parent_key] += 1
        else:
            self._num_needed_children[parent_key] = 1