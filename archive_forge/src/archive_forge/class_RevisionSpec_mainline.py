from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class RevisionSpec_mainline(RevisionIDSpec):
    help_txt = 'Select mainline revision that merged the specified revision.\n\n    Select the revision that merged the specified revision into mainline.\n    '
    prefix = 'mainline:'

    def _as_revision_id(self, context_branch):
        revspec = RevisionSpec.from_string(self.spec)
        if revspec.get_branch() is None:
            spec_branch = context_branch
        else:
            from .branch import Branch
            spec_branch = Branch.open(revspec.get_branch())
        revision_id = revspec.as_revision_id(spec_branch)
        graph = context_branch.repository.get_graph()
        result = graph.find_lefthand_merger(revision_id, context_branch.last_revision())
        if result is None:
            raise InvalidRevisionSpec(self.user_spec, context_branch)
        return result