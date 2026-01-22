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
class WeaveMerger(Merge3Merger):
    """Three-way tree merger, text weave merger."""
    supports_reprocess = True
    supports_show_base = False
    supports_reverse_cherrypick = False
    history_based = True
    requires_file_merge_plan = True

    def _generate_merge_plan(self, this_path, base):
        return self.this_tree.plan_file_merge(this_path, self.other_tree, base=base)

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

    def text_merge(self, trans_id, paths):
        """Perform a (weave) text merge for a given file and file-id.
        If conflicts are encountered, .THIS and .OTHER files will be emitted,
        and a conflict will be noted.
        """
        base_path, other_path, this_path = paths
        lines, base_lines = self._merged_lines(this_path)
        lines = list(lines)
        textfile.check_text_lines(lines)
        self.tt.create_file(lines, trans_id)
        if base_lines is not None:
            self._raw_conflicts.append(('text conflict', trans_id))
            name = self.tt.final_name(trans_id)
            parent_id = self.tt.final_parent(trans_id)
            file_group = self._dump_conflicts(name, paths, parent_id, (base_lines, None, None), no_base=False)
            file_group.append(trans_id)