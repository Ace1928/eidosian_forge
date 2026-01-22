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
def _show_body(self, lf):
    """Show the main log output.

        Subclasses may wish to override this.
        """
    rqst = self.rqst
    if rqst['levels'] is None or lf.get_levels() > rqst['levels']:
        rqst['levels'] = lf.get_levels()
    if not getattr(lf, 'supports_tags', False):
        rqst['generate_tags'] = False
    if not getattr(lf, 'supports_delta', False):
        rqst['delta_type'] = None
    if not getattr(lf, 'supports_diff', False):
        rqst['diff_type'] = None
    if not getattr(lf, 'supports_signatures', False):
        rqst['signature'] = False
    generator = self._generator_factory(self.branch, rqst)
    try:
        for lr in generator.iter_log_revisions():
            lf.log_revision(lr)
    except errors.GhostRevisionUnusableHere:
        raise errors.CommandError(gettext('Further revision history missing.'))
    lf.show_advice()