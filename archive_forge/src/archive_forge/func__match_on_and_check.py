from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
def _match_on_and_check(self, branch, revs):
    info = self._match_on(branch, revs)
    if info:
        return info
    elif info == (None, None):
        return info
    elif self.prefix:
        raise InvalidRevisionSpec(self.user_spec, branch)
    else:
        raise InvalidRevisionSpec(self.spec, branch)