import contextlib
import tempfile
from typing import Type
from .lazy_import import lazy_import
import patiencediff
from breezy import (
from breezy.bzr import (
from breezy.i18n import gettext
from . import decorators, errors, hooks, osutils, registry
from . import revision as _mod_revision
from . import trace, transform
from . import transport as _mod_transport
from . import tree as _mod_tree
@staticmethod
def _subtract_plans(old_plan, new_plan):
    """Remove changes from new_plan that came from old_plan.

        It is assumed that the difference between the old_plan and new_plan
        is their choice of 'b' text.

        All lines from new_plan that differ from old_plan are emitted
        verbatim.  All lines from new_plan that match old_plan but are
        not about the 'b' revision are emitted verbatim.

        Lines that match and are about the 'b' revision are the lines we
        don't want, so we convert 'killed-b' -> 'unchanged', and 'new-b'
        is skipped entirely.
        """
    matcher = patiencediff.PatienceSequenceMatcher(None, old_plan, new_plan)
    last_j = 0
    for i, j, n in matcher.get_matching_blocks():
        for jj in range(last_j, j):
            yield new_plan[jj]
        for jj in range(j, j + n):
            plan_line = new_plan[jj]
            if plan_line[0] == 'new-b':
                pass
            elif plan_line[0] == 'killed-b':
                yield ('unchanged', plan_line[1])
            else:
                yield plan_line
        last_j = j + n