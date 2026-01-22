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
def _merged_lines(self, this_path):
    """Generate the merged lines.
        There is no distinction between lines that are meant to contain <<<<<<<
        and conflicts.
        """
    from .bzr.versionedfile import PlanWeaveMerge
    if self.cherrypick:
        base = self.base_tree
    else:
        base = None
    plan = self._generate_merge_plan(this_path, base)
    if 'merge' in debug.debug_flags:
        plan = list(plan)
        trans_id = self.tt.trans_id_file_id(file_id)
        name = self.tt.final_name(trans_id) + '.plan'
        contents = (b'%11s|%s' % l for l in plan)
        self.tt.new_file(name, self.tt.final_parent(trans_id), contents)
    textmerge = PlanWeaveMerge(plan, b'<<<<<<< TREE\n', b'>>>>>>> MERGE-SOURCE\n')
    lines, conflicts = textmerge.merge_lines(self.reprocess)
    if conflicts:
        base_lines = textmerge.base_from_plan()
    else:
        base_lines = None
    return (lines, base_lines)