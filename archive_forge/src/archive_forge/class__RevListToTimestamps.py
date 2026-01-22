from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
class _RevListToTimestamps:
    """This takes a list of revisions, and allows you to bisect by date"""
    __slots__ = ['branch']

    def __init__(self, branch):
        self.branch = branch

    def __getitem__(self, index):
        """Get the date of the index'd item"""
        r = self.branch.repository.get_revision(self.branch.get_rev_id(index))
        return r.datetime()