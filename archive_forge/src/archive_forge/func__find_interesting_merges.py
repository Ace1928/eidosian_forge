from .. import (
import stat
def _find_interesting_merges(self, commit_refs):
    if commit_refs is None:
        return None
    merges = []
    for commit_ref in commit_refs:
        parent = self._find_interesting_parent(commit_ref)
        if parent is not None:
            merges.append(parent)
    if merges:
        return merges
    else:
        return None