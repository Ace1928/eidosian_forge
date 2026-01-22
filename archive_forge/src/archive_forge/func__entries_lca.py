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
def _entries_lca(self):
    """Gather data about files modified between multiple trees.

        This compares OTHER versus all LCA trees, and for interesting entries,
        it then compares with THIS and BASE.

        For the multi-valued entries, the format will be (BASE, [lca1, lca2])

        :return: [(file_id, changed, paths, parents, names, executable, copied)], where:

            * file_id: Simple file_id of the entry
            * changed: Boolean, True if the kind or contents changed else False
            * paths: ((base, [path, in, lcas]), path_other, path_this)
            * parents: ((base, [parent_id, in, lcas]), parent_id_other,
                        parent_id_this)
            * names:   ((base, [name, in, lcas]), name_in_other, name_in_this)
            * executable: ((base, [exec, in, lcas]), exec_in_other,
                        exec_in_this)
        """
    if self.interesting_files is not None:
        lookup_trees = [self.this_tree, self.base_tree]
        lookup_trees.extend(self._lca_trees)
        interesting_files = self.other_tree.find_related_paths_across_trees(self.interesting_files, lookup_trees)
    else:
        interesting_files = None
    from .multiwalker import MultiWalker
    walker = MultiWalker(self.other_tree, self._lca_trees)
    for other_path, file_id, other_ie, lca_values in walker.iter_all():
        if other_ie is None:
            other_ie = _none_entry
            other_path = None
        if interesting_files is not None and other_path not in interesting_files:
            continue
        is_unmodified = False
        for lca_path, ie in lca_values:
            if ie is not None and other_ie.is_unmodified(ie):
                is_unmodified = True
                break
        if is_unmodified:
            continue
        lca_entries = []
        lca_paths = []
        for lca_path, lca_ie in lca_values:
            if lca_ie is None:
                lca_entries.append(_none_entry)
                lca_paths.append(None)
            else:
                lca_entries.append(lca_ie)
                lca_paths.append(lca_path)
        try:
            base_path = self.base_tree.id2path(file_id)
        except errors.NoSuchId:
            base_path = None
            base_ie = _none_entry
        else:
            base_ie = next(self.base_tree.iter_entries_by_dir(specific_files=[base_path]))[1]
        try:
            this_path = self.this_tree.id2path(file_id)
        except errors.NoSuchId:
            this_ie = _none_entry
            this_path = None
        else:
            this_ie = next(self.this_tree.iter_entries_by_dir(specific_files=[this_path]))[1]
        lca_kinds = []
        lca_parent_ids = []
        lca_names = []
        lca_executable = []
        for lca_ie in lca_entries:
            lca_kinds.append(lca_ie.kind)
            lca_parent_ids.append(lca_ie.parent_id)
            lca_names.append(lca_ie.name)
            lca_executable.append(lca_ie.executable)
        kind_winner = self._lca_multi_way((base_ie.kind, lca_kinds), other_ie.kind, this_ie.kind)
        parent_id_winner = self._lca_multi_way((base_ie.parent_id, lca_parent_ids), other_ie.parent_id, this_ie.parent_id)
        name_winner = self._lca_multi_way((base_ie.name, lca_names), other_ie.name, this_ie.name)
        content_changed = True
        if kind_winner == 'this':
            if other_ie.kind == 'directory':
                if parent_id_winner == 'this' and name_winner == 'this':
                    continue
                content_changed = False
            elif other_ie.kind is None or other_ie.kind == 'file':

                def get_sha1(tree, path):
                    if path is None:
                        return None
                    try:
                        return tree.get_file_sha1(path)
                    except _mod_transport.NoSuchFile:
                        return None
                base_sha1 = get_sha1(self.base_tree, base_path)
                lca_sha1s = [get_sha1(tree, lca_path) for tree, lca_path in zip(self._lca_trees, lca_paths)]
                this_sha1 = get_sha1(self.this_tree, this_path)
                other_sha1 = get_sha1(self.other_tree, other_path)
                sha1_winner = self._lca_multi_way((base_sha1, lca_sha1s), other_sha1, this_sha1, allow_overriding_lca=False)
                exec_winner = self._lca_multi_way((base_ie.executable, lca_executable), other_ie.executable, this_ie.executable)
                if parent_id_winner == 'this' and name_winner == 'this' and (sha1_winner == 'this') and (exec_winner == 'this'):
                    continue
                if sha1_winner == 'this':
                    content_changed = False
            elif other_ie.kind == 'symlink':

                def get_target(ie, tree, path):
                    if ie.kind != 'symlink':
                        return None
                    return tree.get_symlink_target(path)
                base_target = get_target(base_ie, self.base_tree, base_path)
                lca_targets = [get_target(ie, tree, lca_path) for ie, tree, lca_path in zip(lca_entries, self._lca_trees, lca_paths)]
                this_target = get_target(this_ie, self.this_tree, this_path)
                other_target = get_target(other_ie, self.other_tree, other_path)
                target_winner = self._lca_multi_way((base_target, lca_targets), other_target, this_target)
                if parent_id_winner == 'this' and name_winner == 'this' and (target_winner == 'this'):
                    continue
                if target_winner == 'this':
                    content_changed = False
            elif other_ie.kind == 'tree-reference':
                content_changed = False
                if parent_id_winner == 'this' and name_winner == 'this':
                    continue
            else:
                raise AssertionError('unhandled kind: %s' % other_ie.kind)
        yield (file_id, content_changed, ((base_path, lca_paths), other_path, this_path), ((base_ie.parent_id, lca_parent_ids), other_ie.parent_id, this_ie.parent_id), ((base_ie.name, lca_names), other_ie.name, this_ie.name), ((base_ie.executable, lca_executable), other_ie.executable, this_ie.executable), False)