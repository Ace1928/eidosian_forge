from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class RevisionSpec_tag(RevisionSpec):
    """Select a revision identified by tag name"""
    help_txt = "Selects a revision identified by a tag name.\n\n    Tags are stored in the branch and created by the 'tag' command.\n    "
    prefix = 'tag:'
    dwim_catchable_exceptions = [errors.NoSuchTag, errors.TagsNotSupported]

    def _match_on(self, branch, revs):
        return RevisionInfo.from_revision_id(branch, branch.tags.lookup_tag(self.spec))

    def _as_revision_id(self, context_branch):
        return context_branch.tags.lookup_tag(self.spec)