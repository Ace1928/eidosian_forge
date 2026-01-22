import contextlib
import errno
import os
import tempfile
import time
from stat import S_IEXEC, S_ISREG
from .. import (annotate, conflicts, controldir, errors, lock, multiparent,
from .. import revision as _mod_revision
from .. import trace
from .. import transport as _mod_transport
from .. import tree, ui, urlutils
from ..filters import ContentFilterContext, filtered_output_bytes
from ..i18n import gettext
from ..mutabletree import MutableTree
from ..progress import ProgressPhase
from ..transform import (ROOT_PARENT, FinalPaths, ImmortalLimbo,
from ..tree import find_previous_path
from . import inventory, inventorytree
from .conflicts import Conflict
def _build_tree(tree, wt, accelerator_tree, hardlink, delta_from_tree):
    """See build_tree."""
    for num, _unused in enumerate(wt.all_versioned_paths()):
        if num > 0:
            raise errors.WorkingTreeAlreadyPopulated(base=wt.basedir)
    file_trans_id = {}
    top_pb = ui.ui_factory.nested_progress_bar()
    pp = ProgressPhase('Build phase', 2, top_pb)
    if tree.path2id('') is not None:
        if wt.path2id('') != tree.path2id(''):
            wt.set_root_id(tree.path2id(''))
            wt.flush()
    tt = wt.transform()
    divert = set()
    try:
        pp.next_phase()
        file_trans_id[find_previous_path(wt, tree, '')] = tt.trans_id_tree_path('')
        with ui.ui_factory.nested_progress_bar() as pb:
            deferred_contents = []
            num = 0
            total = len(tree.all_versioned_paths())
            if delta_from_tree:
                precomputed_delta = []
            else:
                precomputed_delta = None
            if total > 0:
                existing_files = set()
                for dir, files in wt.walkdirs():
                    existing_files.update((f[0] for f in files))
            for num, (tree_path, entry) in enumerate(tree.iter_entries_by_dir()):
                pb.update(gettext('Building tree'), num - len(deferred_contents), total)
                if entry.parent_id is None:
                    continue
                reparent = False
                file_id = entry.file_id
                if delta_from_tree:
                    precomputed_delta.append((None, tree_path, file_id, entry))
                if tree_path in existing_files:
                    target_path = wt.abspath(tree_path)
                    kind = osutils.file_kind(target_path)
                    if kind == 'directory':
                        try:
                            controldir.ControlDir.open(target_path)
                        except errors.NotBranchError:
                            pass
                        else:
                            divert.add(tree_path)
                    if tree_path not in divert and _content_match(tree, entry, tree_path, kind, target_path):
                        tt.delete_contents(tt.trans_id_tree_path(tree_path))
                        if kind == 'directory':
                            reparent = True
                parent_id = file_trans_id[osutils.dirname(tree_path)]
                if entry.kind == 'file':
                    trans_id = tt.create_path(entry.name, parent_id)
                    file_trans_id[tree_path] = trans_id
                    tt.version_file(trans_id, file_id=file_id)
                    executable = tree.is_executable(tree_path)
                    if executable:
                        tt.set_executability(executable, trans_id)
                    trans_data = (trans_id, tree_path, entry.text_sha1)
                    deferred_contents.append((tree_path, trans_data))
                else:
                    file_trans_id[tree_path] = new_by_entry(tree_path, tt, entry, parent_id, tree)
                if reparent:
                    new_trans_id = file_trans_id[tree_path]
                    old_parent = tt.trans_id_tree_path(tree_path)
                    _reparent_children(tt, old_parent, new_trans_id)
            offset = num + 1 - len(deferred_contents)
            _create_files(tt, tree, deferred_contents, pb, offset, accelerator_tree, hardlink)
        pp.next_phase()
        divert_trans = {file_trans_id[f] for f in divert}

        def resolver(t, c):
            return resolve_checkout(t, c, divert_trans)
        raw_conflicts = resolve_conflicts(tt, pass_func=resolver)
        if len(raw_conflicts) > 0:
            precomputed_delta = None
        conflicts = tt.cook_conflicts(raw_conflicts)
        for conflict in conflicts:
            trace.warning(str(conflict))
        try:
            wt.add_conflicts(conflicts)
        except errors.UnsupportedOperation:
            pass
        result = tt.apply(no_conflicts=True, precomputed_delta=precomputed_delta)
    finally:
        tt.finalize()
        top_pb.finished()
    return result