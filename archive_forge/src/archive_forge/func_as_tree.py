from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
def as_tree(self, context_branch):
    """Return the tree object for this revisions spec.

        Some revision specs require a context_branch to be able to determine
        the revision id and access the repository. Not all specs will make
        use of it.
        """
    return self._as_tree(context_branch)