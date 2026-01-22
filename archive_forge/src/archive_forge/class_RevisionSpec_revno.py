from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class RevisionSpec_revno(RevisionSpec):
    """Selects a revision using a number."""
    help_txt = "Selects a revision using a number.\n\n    Use an integer to specify a revision in the history of the branch.\n    Optionally a branch can be specified.  A negative number will count\n    from the end of the branch (-1 is the last revision, -2 the previous\n    one). If the negative number is larger than the branch's history, the\n    first revision is returned.\n    Examples::\n\n      revno:1                   -> return the first revision of this branch\n      revno:3:/path/to/branch   -> return the 3rd revision of\n                                   the branch '/path/to/branch'\n      revno:-1                  -> The last revision in a branch.\n      -2:http://other/branch    -> The second to last revision in the\n                                   remote branch.\n      -1000000                  -> Most likely the first revision, unless\n                                   your history is very long.\n    "
    prefix = 'revno:'

    def _match_on(self, branch, revs):
        """Lookup a revision by revision number"""
        branch, revno, revision_id = self._lookup(branch)
        return RevisionInfo(branch, revno, revision_id)

    def _lookup(self, branch):
        loc = self.spec.find(':')
        if loc == -1:
            revno_spec = self.spec
            branch_spec = None
        else:
            revno_spec = self.spec[:loc]
            branch_spec = self.spec[loc + 1:]
        if revno_spec == '':
            if not branch_spec:
                raise InvalidRevisionSpec(self.user_spec, branch, 'cannot have an empty revno and no branch')
            revno = None
        else:
            try:
                revno = int(revno_spec)
                dotted = False
            except ValueError:
                try:
                    match_revno = tuple((int(number) for number in revno_spec.split('.')))
                except ValueError as e:
                    raise InvalidRevisionSpec(self.user_spec, branch, e)
                dotted = True
        if branch_spec:
            from .branch import Branch
            branch = Branch.open(branch_spec)
        if dotted:
            try:
                revision_id = branch.dotted_revno_to_revision_id(match_revno, _cache_reverse=True)
            except (errors.NoSuchRevision, errors.RevnoOutOfBounds):
                raise InvalidRevisionSpec(self.user_spec, branch)
            else:
                return (branch, None, revision_id)
        else:
            last_revno, last_revision_id = branch.last_revision_info()
            if revno < 0:
                if -revno >= last_revno:
                    revno = 1
                else:
                    revno = last_revno + revno + 1
            try:
                revision_id = branch.get_rev_id(revno)
            except (errors.NoSuchRevision, errors.RevnoOutOfBounds):
                raise InvalidRevisionSpec(self.user_spec, branch)
        return (branch, revno, revision_id)

    def _as_revision_id(self, context_branch):
        branch, revno, revision_id = self._lookup(context_branch)
        return revision_id

    def needs_branch(self):
        return self.spec.find(':') == -1

    def get_branch(self):
        if self.spec.find(':') == -1:
            return None
        else:
            return self.spec[self.spec.find(':') + 1:]