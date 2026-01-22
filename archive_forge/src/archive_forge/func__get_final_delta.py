from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _get_final_delta(self):
    """Generate the final delta.

        Smart post-processing of changes, e.g. pruning of directories
        that would become empty, goes here.
        """
    delta = list(self._delta_entries_by_fileid.values())
    if self.prune_empty_dirs and self._dirs_that_might_become_empty:
        candidates = self._dirs_that_might_become_empty
        while candidates:
            never_born = set()
            parent_dirs_that_might_become_empty = set()
            for path, file_id in self._empty_after_delta(delta, candidates):
                newly_added = self._new_file_ids.get(path)
                if newly_added:
                    never_born.add(newly_added)
                else:
                    delta.append((path, None, file_id, None))
                parent_dir = osutils.dirname(path)
                if parent_dir:
                    parent_dirs_that_might_become_empty.add(parent_dir)
            candidates = parent_dirs_that_might_become_empty
            if never_born:
                delta = [de for de in delta if de[2] not in never_born]
    return delta