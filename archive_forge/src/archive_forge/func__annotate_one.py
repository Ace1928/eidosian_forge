from . import errors
from . import graph as _mod_graph
from . import osutils, ui
def _annotate_one(self, key, text, num_lines):
    this_annotation = (key,)
    annotations = [this_annotation] * num_lines
    parent_keys = self._parent_map[key]
    if parent_keys:
        self._update_from_first_parent(key, annotations, text, parent_keys[0])
        for parent in parent_keys[1:]:
            self._update_from_other_parents(key, annotations, text, this_annotation, parent)
    self._record_annotation(key, parent_keys, annotations)