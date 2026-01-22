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
class Merge3Merger:
    """Three-way merger that uses the merge3 text merger"""
    requires_base = True
    supports_reprocess = True
    supports_show_base = True
    history_based = False
    supports_cherrypick = True
    supports_reverse_cherrypick = True
    winner_idx = {'this': 2, 'other': 1, 'conflict': 1}
    supports_lca_trees = True
    requires_file_merge_plan = False

    def __init__(self, working_tree, this_tree, base_tree, other_tree, reprocess=False, show_base=False, change_reporter=None, interesting_files=None, do_merge=True, cherrypick=False, lca_trees=None, this_branch=None, other_branch=None):
        """Initialize the merger object and perform the merge.

        :param working_tree: The working tree to apply the merge to
        :param this_tree: The local tree in the merge operation
        :param base_tree: The common tree in the merge operation
        :param other_tree: The other tree to merge changes from
        :param this_branch: The branch associated with this_tree.  Defaults to
            this_tree.branch if not supplied.
        :param other_branch: The branch associated with other_tree, if any.
        :param: reprocess If True, perform conflict-reduction processing.
        :param show_base: If True, show the base revision in text conflicts.
            (incompatible with reprocess)
        :param change_reporter: An object that should report changes made
        :param interesting_files: The tree-relative paths of files that should
            participate in the merge.  If these paths refer to directories,
            the contents of those directories will also be included.  If not
            specified, all files may participate in the
            merge.
        :param lca_trees: Can be set to a dictionary of {revision_id:rev_tree}
            if the ancestry was found to include a criss-cross merge.
            Otherwise should be None.
        """
        object.__init__(self)
        if this_branch is None:
            this_branch = this_tree.branch
        self.interesting_files = interesting_files
        self.working_tree = working_tree
        self.this_tree = this_tree
        self.base_tree = base_tree
        self.other_tree = other_tree
        self.this_branch = this_branch
        self.other_branch = other_branch
        self._raw_conflicts = []
        self.cooked_conflicts = []
        self.reprocess = reprocess
        self.show_base = show_base
        self._lca_trees = lca_trees
        self.change_reporter = change_reporter
        self.cherrypick = cherrypick
        if do_merge:
            self.do_merge()

    def do_merge(self):
        with contextlib.ExitStack() as stack:
            stack.enter_context(self.working_tree.lock_tree_write())
            stack.enter_context(self.this_tree.lock_read())
            stack.enter_context(self.base_tree.lock_read())
            stack.enter_context(self.other_tree.lock_read())
            self.tt = self.working_tree.transform()
            stack.enter_context(self.tt)
            self._compute_transform()
            results = self.tt.apply(no_conflicts=True)
            self.write_modified(results)
            try:
                self.working_tree.add_conflicts(self.cooked_conflicts)
            except errors.UnsupportedOperation:
                pass

    def make_preview_transform(self):
        with self.base_tree.lock_read(), self.other_tree.lock_read():
            self.tt = self.working_tree.preview_transform()
            self._compute_transform()
            return self.tt

    def _compute_transform(self):
        if self._lca_trees is None:
            entries = list(self._entries3())
            resolver = self._three_way
        else:
            entries = list(self._entries_lca())
            resolver = self._lca_multi_way
        factories = Merger.hooks['merge_file_content']
        hooks = [factory(self) for factory in factories] + [self]
        self.active_hooks = [hook for hook in hooks if hook is not None]
        with ui.ui_factory.nested_progress_bar() as child_pb:
            for num, (file_id, changed, paths3, parents3, names3, executable3, copied) in enumerate(entries):
                if copied:
                    paths3 = (None, paths3[1], None)
                    parents3 = (None, parents3[1], None)
                    names3 = (None, names3[1], None)
                    executable3 = (None, executable3[1], None)
                    changed = True
                    copied = False
                trans_id = self.tt.trans_id_file_id(file_id)
                child_pb.update(gettext('Preparing file merge'), num, len(entries))
                self._merge_names(trans_id, file_id, paths3, parents3, names3, resolver=resolver)
                if changed:
                    file_status = self._do_merge_contents(paths3, trans_id, file_id)
                else:
                    file_status = 'unmodified'
                self._merge_executable(paths3, trans_id, executable3, file_status, resolver=resolver)
        self.tt.fixup_new_roots()
        self._finish_computing_transform()

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

    def _entries3(self):
        """Gather data about files modified between three trees.

        Return a list of tuples of file_id, changed, parents3, names3,
        executable3.  changed is a boolean indicating whether the file contents
        or kind were changed.  parents3 is a tuple of parent ids for base,
        other and this.  names3 is a tuple of names for base, other and this.
        executable3 is a tuple of execute-bit values for base, other and this.
        """
        iterator = self.other_tree.iter_changes(self.base_tree, specific_files=self.interesting_files, extra_trees=[self.this_tree])
        this_interesting_files = self.this_tree.find_related_paths_across_trees(self.interesting_files, trees=[self.other_tree])
        this_entries = dict(self.this_tree.iter_entries_by_dir(specific_files=this_interesting_files))
        for change in iterator:
            if change.path[0] is not None:
                this_path = _mod_tree.find_previous_path(self.base_tree, self.this_tree, change.path[0])
            else:
                this_path = _mod_tree.find_previous_path(self.other_tree, self.this_tree, change.path[1])
            this_entry = this_entries.get(this_path)
            if this_entry is not None:
                this_name = this_entry.name
                this_parent = this_entry.parent_id
                this_executable = this_entry.executable
            else:
                this_name = None
                this_parent = None
                this_executable = None
            parents3 = change.parent_id + (this_parent,)
            names3 = change.name + (this_name,)
            paths3 = change.path + (this_path,)
            executable3 = change.executable + (this_executable,)
            yield (change.file_id, change.changed_content, paths3, parents3, names3, executable3, change.copied)

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

    def write_modified(self, results):
        if not self.working_tree.supports_merge_modified():
            return
        modified_hashes = {}
        for path in results.modified_paths:
            wt_relpath = self.working_tree.relpath(path)
            if not self.working_tree.is_versioned(wt_relpath):
                continue
            hash = self.working_tree.get_file_sha1(wt_relpath)
            if hash is None:
                continue
            modified_hashes[wt_relpath] = hash
        self.working_tree.set_merge_modified(modified_hashes)

    @staticmethod
    def parent(entry):
        """Determine the parent for a file_id (used as a key method)"""
        if entry is None:
            return None
        return entry.parent_id

    @staticmethod
    def name(entry):
        """Determine the name for a file_id (used as a key method)"""
        if entry is None:
            return None
        return entry.name

    @staticmethod
    def contents_sha1(tree, path):
        """Determine the sha1 of the file contents (used as a key method)."""
        try:
            return tree.get_file_sha1(path)
        except _mod_transport.NoSuchFile:
            return None

    @staticmethod
    def executable(tree, path):
        """Determine the executability of a file-id (used as a key method)."""
        try:
            if tree.kind(path) != 'file':
                return False
        except _mod_transport.NoSuchFile:
            return None
        return tree.is_executable(path)

    @staticmethod
    def kind(tree, path):
        """Determine the kind of a file-id (used as a key method)."""
        try:
            return tree.kind(path)
        except _mod_transport.NoSuchFile:
            return None

    @staticmethod
    def _three_way(base, other, this):
        if base == other:
            return 'this'
        elif this not in (base, other):
            return 'conflict'
        elif this == other:
            return 'this'
        else:
            return 'other'

    @staticmethod
    def _lca_multi_way(bases, other, this, allow_overriding_lca=True):
        """Consider LCAs when determining whether a change has occurred.

        If LCAS are all identical, this is the same as a _three_way comparison.

        :param bases: value in (BASE, [LCAS])
        :param other: value in OTHER
        :param this: value in THIS
        :param allow_overriding_lca: If there is more than one unique lca
            value, allow OTHER to override THIS if it has a new value, and
            THIS only has an lca value, or vice versa. This is appropriate for
            truly scalar values, not as much for non-scalars.
        :return: 'this', 'other', or 'conflict' depending on whether an entry
            changed or not.
        """
        if other == this:
            return 'this'
        base_val, lca_vals = bases
        filtered_lca_vals = [lca_val for lca_val in lca_vals if lca_val != base_val]
        if len(filtered_lca_vals) == 0:
            return Merge3Merger._three_way(base_val, other, this)
        unique_lca_vals = set(filtered_lca_vals)
        if len(unique_lca_vals) == 1:
            return Merge3Merger._three_way(unique_lca_vals.pop(), other, this)
        if allow_overriding_lca:
            if other in unique_lca_vals:
                if this in unique_lca_vals:
                    return 'conflict'
                else:
                    return 'this'
            elif this in unique_lca_vals:
                return 'other'
        return 'conflict'

    def _merge_names(self, trans_id, file_id, paths, parents, names, resolver):
        """Perform a merge on file names and parents"""
        base_name, other_name, this_name = names
        base_parent, other_parent, this_parent = parents
        unused_base_path, other_path, this_path = paths
        name_winner = resolver(*names)
        parent_id_winner = resolver(*parents)
        if this_name is None:
            if name_winner == 'this':
                name_winner = 'other'
            if parent_id_winner == 'this':
                parent_id_winner = 'other'
        if name_winner == 'this' and parent_id_winner == 'this':
            return
        if name_winner == 'conflict' or parent_id_winner == 'conflict':
            self._raw_conflicts.append(('path conflict', trans_id, file_id, this_parent, this_name, other_parent, other_name))
        if other_path is None:
            return
        parent_id = parents[self.winner_idx[parent_id_winner]]
        name = names[self.winner_idx[name_winner]]
        if parent_id is not None or name is not None:
            if parent_id is None and name is not None:
                if names[self.winner_idx[parent_id_winner]] != '':
                    raise AssertionError('File looks like a root, but named %s' % names[self.winner_idx[parent_id_winner]])
                parent_trans_id = transform.ROOT_PARENT
            else:
                parent_trans_id = self.tt.trans_id_file_id(parent_id)
            self.tt.adjust_path(name, parent_trans_id, trans_id)

    def _do_merge_contents(self, paths, trans_id, file_id):
        """Performs a merge on file_id contents."""

        def contents_pair(tree, path):
            if path is None:
                return (None, None)
            try:
                kind = tree.kind(path)
            except _mod_transport.NoSuchFile:
                return (None, None)
            if kind == 'file':
                contents = tree.get_file_sha1(path)
            elif kind == 'symlink':
                contents = tree.get_symlink_target(path)
            else:
                contents = None
            return (kind, contents)
        base_path, other_path, this_path = paths
        other_pair = contents_pair(self.other_tree, other_path)
        this_pair = contents_pair(self.this_tree, this_path)
        if self._lca_trees:
            base_path, lca_paths = base_path
            base_pair = contents_pair(self.base_tree, base_path)
            lca_pairs = [contents_pair(tree, path) for tree, path in zip(self._lca_trees, lca_paths)]
            winner = self._lca_multi_way((base_pair, lca_pairs), other_pair, this_pair, allow_overriding_lca=False)
        else:
            base_pair = contents_pair(self.base_tree, base_path)
            if base_pair == other_pair:
                winner = 'this'
            else:
                this_pair = contents_pair(self.this_tree, this_path)
                winner = self._three_way(base_pair, other_pair, this_pair)
        if winner == 'this':
            return 'unmodified'
        params = MergeFileHookParams(self, (base_path, other_path, this_path), trans_id, this_pair[0], other_pair[0], winner)
        hooks = self.active_hooks
        hook_status = 'not_applicable'
        for hook in hooks:
            hook_status, lines = hook.merge_contents(params)
            if hook_status != 'not_applicable':
                break
        keep_this = False
        result = 'modified'
        if hook_status == 'not_applicable':
            result = None
            name = self.tt.final_name(trans_id)
            parent_id = self.tt.final_parent(trans_id)
            inhibit_content_conflict = False
            if params.this_kind is None:
                if self.this_tree.is_versioned(other_path):
                    keep_this = True
                    self.tt.version_file(trans_id, file_id=file_id)
                    transform.create_from_tree(self.tt, trans_id, self.other_tree, other_path, filter_tree_path=self._get_filter_tree_path(other_path))
                    inhibit_content_conflict = True
            elif params.other_kind is None:
                if self.other_tree.is_versioned(this_path):
                    keep_this = True
                    inhibit_content_conflict = True
            if not inhibit_content_conflict:
                if params.this_kind is not None:
                    self.tt.unversion_file(trans_id)
                file_group = self._dump_conflicts(name, (base_path, other_path, this_path), parent_id)
                for tid in file_group:
                    self.tt.version_file(tid, file_id=file_id)
                    break
                self._raw_conflicts.append(('contents conflict', file_group))
        elif hook_status == 'success':
            self.tt.create_file(lines, trans_id)
        elif hook_status == 'conflicted':
            self.tt.create_file(lines, trans_id)
            self._raw_conflicts.append(('text conflict', trans_id))
            name = self.tt.final_name(trans_id)
            parent_id = self.tt.final_parent(trans_id)
            self._dump_conflicts(name, (base_path, other_path, this_path), parent_id)
        elif hook_status == 'delete':
            self.tt.unversion_file(trans_id)
            result = 'deleted'
        elif hook_status == 'done':
            pass
        else:
            raise AssertionError('unknown hook_status: {!r}'.format(hook_status))
        if not this_path and result == 'modified':
            self.tt.version_file(trans_id, file_id=file_id)
        if not keep_this:
            self.tt.delete_contents(trans_id)
        return result

    def _default_other_winner_merge(self, merge_hook_params):
        """Replace this contents with other."""
        trans_id = merge_hook_params.trans_id
        if merge_hook_params.other_path is not None:
            transform.create_from_tree(self.tt, trans_id, self.other_tree, merge_hook_params.other_path, filter_tree_path=self._get_filter_tree_path(merge_hook_params.other_path))
            return ('done', None)
        elif merge_hook_params.this_path is not None:
            return ('delete', None)
        else:
            raise AssertionError('winner is OTHER, but file %r not in THIS or OTHER tree' % (merge_hook_params.base_path,))

    def merge_contents(self, merge_hook_params):
        """Fallback merge logic after user installed hooks."""
        if merge_hook_params.winner == 'other':
            return self._default_other_winner_merge(merge_hook_params)
        elif merge_hook_params.is_file_merge():
            try:
                self.text_merge(merge_hook_params.trans_id, merge_hook_params.paths)
            except errors.BinaryFile:
                return ('not_applicable', None)
            return ('done', None)
        else:
            return ('not_applicable', None)

    def get_lines(self, tree, path):
        """Return the lines in a file, or an empty list."""
        if path is None:
            return []
        try:
            kind = tree.kind(path)
        except _mod_transport.NoSuchFile:
            return []
        else:
            if kind != 'file':
                return []
            return tree.get_file_lines(path)

    def text_merge(self, trans_id, paths):
        """Perform a three-way text merge on a file"""
        from merge3 import Merge3
        base_path, other_path, this_path = paths
        base_lines = self.get_lines(self.base_tree, base_path)
        other_lines = self.get_lines(self.other_tree, other_path)
        this_lines = self.get_lines(self.this_tree, this_path)
        textfile.check_text_lines(base_lines)
        textfile.check_text_lines(other_lines)
        textfile.check_text_lines(this_lines)
        m3 = Merge3(base_lines, this_lines, other_lines, is_cherrypick=self.cherrypick, sequence_matcher=patiencediff.PatienceSequenceMatcher)
        start_marker = b'!START OF MERGE CONFLICT!' + b'I HOPE THIS IS UNIQUE'
        if self.show_base is True:
            base_marker = b'|' * 7
        else:
            base_marker = None

        def iter_merge3(retval):
            retval['text_conflicts'] = False
            if base_marker and self.reprocess:
                raise CantReprocessAndShowBase()
            lines = list(m3.merge_lines(name_a=b'TREE', name_b=b'MERGE-SOURCE', name_base=b'BASE-REVISION', start_marker=start_marker, base_marker=base_marker, reprocess=self.reprocess))
            for line in lines:
                if line.startswith(start_marker):
                    retval['text_conflicts'] = True
                    yield line.replace(start_marker, b'<' * 7)
                else:
                    yield line
        retval = {}
        merge3_iterator = iter_merge3(retval)
        self.tt.create_file(merge3_iterator, trans_id)
        if retval['text_conflicts'] is True:
            self._raw_conflicts.append(('text conflict', trans_id))
            name = self.tt.final_name(trans_id)
            parent_id = self.tt.final_parent(trans_id)
            file_group = self._dump_conflicts(name, paths, parent_id, lines=(base_lines, other_lines, this_lines))
            file_group.append(trans_id)

    def _get_filter_tree_path(self, path):
        if self.this_tree.supports_content_filtering():
            filter_path = _mod_tree.find_previous_path(self.other_tree, self.working_tree, path)
            if filter_path is None:
                filter_path = path
            return filter_path
        return None

    def _dump_conflicts(self, name, paths, parent_id, lines=None, no_base=False):
        """Emit conflict files.
        If this_lines, base_lines, or other_lines are omitted, they will be
        determined automatically.  If set_version is true, the .OTHER, .THIS
        or .BASE (in that order) will be created as versioned files.
        """
        base_path, other_path, this_path = paths
        if lines:
            base_lines, other_lines, this_lines = lines
        else:
            base_lines = other_lines = this_lines = None
        data = [('OTHER', self.other_tree, other_path, other_lines), ('THIS', self.this_tree, this_path, this_lines)]
        if not no_base:
            data.append(('BASE', self.base_tree, base_path, base_lines))
        if self.this_tree.supports_content_filtering():
            filter_tree_path = this_path
        else:
            filter_tree_path = None
        file_group = []
        for suffix, tree, path, lines in data:
            if path is not None:
                trans_id = self._conflict_file(name, parent_id, path, tree, suffix, lines, filter_tree_path)
                file_group.append(trans_id)
        return file_group

    def _conflict_file(self, name, parent_id, path, tree, suffix, lines=None, filter_tree_path=None):
        """Emit a single conflict file."""
        name = name + '.' + suffix
        trans_id = self.tt.create_path(name, parent_id)
        transform.create_from_tree(self.tt, trans_id, tree, path, chunks=lines, filter_tree_path=filter_tree_path)
        return trans_id

    def _merge_executable(self, paths, trans_id, executable, file_status, resolver):
        """Perform a merge on the execute bit."""
        base_executable, other_executable, this_executable = executable
        base_path, other_path, this_path = paths
        if file_status == 'deleted':
            return
        winner = resolver(*executable)
        if winner == 'conflict':
            if other_path is None:
                winner = 'this'
            else:
                winner = 'other'
        if winner == 'this' and file_status != 'modified':
            return
        if self.tt.final_kind(trans_id) != 'file':
            return
        if winner == 'this':
            executability = this_executable
        elif other_path is not None:
            executability = other_executable
        elif this_path is not None:
            executability = this_executable
        elif base_path is not None:
            executability = base_executable
        if executability is not None:
            self.tt.set_executability(executability, trans_id)

    def cook_conflicts(self, fs_conflicts):
        """Convert all conflicts into a form that doesn't depend on trans_id"""
        self.cooked_conflicts = list(self.tt.cook_conflicts(list(fs_conflicts) + self._raw_conflicts))