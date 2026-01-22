import _imp
import _io
import sys
import _warnings
import marshal
def _recalculate(self):
    parent_path = tuple(self._get_parent_path())
    if parent_path != self._last_parent_path or self._epoch != self._last_epoch:
        spec = self._path_finder(self._name, parent_path)
        if spec is not None and spec.loader is None:
            if spec.submodule_search_locations:
                self._path = spec.submodule_search_locations
        self._last_parent_path = parent_path
        self._last_epoch = self._epoch
    return self._path