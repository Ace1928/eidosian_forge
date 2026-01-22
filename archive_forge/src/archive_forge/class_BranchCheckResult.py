from typing import (TYPE_CHECKING, Dict, List, Optional, TextIO, Tuple, Union,
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
import contextlib
import itertools
from . import config as _mod_config
from . import debug, errors, registry, repository
from . import revision as _mod_revision
from . import urlutils
from .controldir import (ControlComponent, ControlComponentFormat,
from .hooks import Hooks
from .inter import InterObject
from .lock import LogicalLockResult
from .revision import RevisionID
from .trace import is_quiet, mutter, mutter_callsite, note, warning
from .transport import Transport, get_transport
class BranchCheckResult:
    """Results of checking branch consistency.

    See `Branch.check`
    """

    def __init__(self, branch):
        self.branch = branch
        self.errors = []

    def report_results(self, verbose: bool) -> None:
        """Report the check results via trace.note.

        Args:
          verbose: Requests more detailed display of what was checked,
            if any.
        """
        from breezy.i18n import gettext, ngettext
        note(gettext('checked branch {0} format {1}').format(self.branch.user_url, self.branch._format))
        for error in self.errors:
            note(gettext('found error:%s'), error)