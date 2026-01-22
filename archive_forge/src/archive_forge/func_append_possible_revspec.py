from typing import List, Optional, Type
from breezy import revision, workingtree
from breezy.i18n import gettext
from . import errors, lazy_regex, registry
from . import revision as _mod_revision
from . import trace
@classmethod
def append_possible_revspec(cls, revspec):
    """Append a possible DWIM revspec.

        :param revspec: Revision spec to try.
        """
    cls._possible_revspecs.append(registry._ObjectGetter(revspec))