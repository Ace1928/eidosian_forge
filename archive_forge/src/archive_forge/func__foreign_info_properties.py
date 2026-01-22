import codecs
import itertools
import re
import sys
from io import BytesIO
from typing import Callable, Dict, List
from warnings import warn
from .lazy_import import lazy_import
from breezy import (
from breezy.i18n import gettext, ngettext
from . import errors, registry
from . import revision as _mod_revision
from . import revisionspec, trace
from . import transport as _mod_transport
from .osutils import (format_date,
from .tree import InterTree, find_previous_path
def _foreign_info_properties(self, rev):
    """Custom log displayer for foreign revision identifiers.

        :param rev: Revision object.
        """
    if isinstance(rev, foreign.ForeignRevision):
        return self._format_properties(rev.mapping.vcs.show_foreign_revid(rev.foreign_revid))
    if b':' not in rev.revision_id:
        return []
    try:
        foreign_revid, mapping = foreign.foreign_vcs_registry.parse_revision_id(rev.revision_id)
    except errors.InvalidRevisionId:
        return []
    return self._format_properties(mapping.vcs.show_foreign_revid(foreign_revid))