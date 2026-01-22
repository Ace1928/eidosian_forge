from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
def as_revision_id(self, context_branch):
    """Return just the revision_id for this revisions spec.

        Some revision specs require a context_branch to be able to determine
        their value. Not all specs will make use of it.
        """
    return self._as_revision_id(context_branch)