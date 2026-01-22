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
def _finish_computing_transform(self):
    """Finalize the transform and report the changes.

        This is the second half of _compute_transform.
        """
    with ui.ui_factory.nested_progress_bar() as child_pb:
        fs_conflicts = transform.resolve_conflicts(self.tt, child_pb, lambda t, c: transform.conflict_pass(t, c, self.other_tree))
    if self.change_reporter is not None:
        from breezy import delta
        delta.report_changes(self.tt.iter_changes(), self.change_reporter)
    self.cook_conflicts(fs_conflicts)
    for conflict in self.cooked_conflicts:
        trace.warning('%s', conflict.describe())