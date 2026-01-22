from .. import (
import stat
def _find_interesting_parent(self, commit_ref):
    while True:
        if commit_ref not in self.squashed_commits:
            return commit_ref
        parents = self.parents.get(commit_ref)
        if not parents:
            return None
        commit_ref = parents[0]