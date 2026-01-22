import os
import sys
import tempfile
import breezy
from .. import controldir, errors, merge_directive, osutils
from ..bzr import generate_ids
from ..bzr.conflicts import ContentsConflict, PathConflict, TextConflict
from ..merge import Diff3Merger, Merge3Merger, Merger, WeaveMerger
from ..osutils import getcwd, pathjoin
from ..workingtree import WorkingTree
from . import TestCaseWithTransport, TestSkipped, features
class MergeBuilder:

    def __init__(self, dir=None):
        self.dir = tempfile.mkdtemp(prefix='merge-test', dir=dir)
        self.tree_root = generate_ids.gen_root_id()

        def wt(name):
            path = pathjoin(self.dir, name)
            os.mkdir(path)
            wt = controldir.ControlDir.create_standalone_workingtree(path)
            wt.lock_write()
            if wt.supports_file_ids:
                wt.set_root_id(self.tree_root)
            wt.flush()
            tt = wt.transform()
            return (wt, tt)
        self.base, self.base_tt = wt('base')
        self.this, self.this_tt = wt('this')
        self.other, self.other_tt = wt('other')

    def root(self):
        ret = []
        for tt in [self.this_tt, self.base_tt, self.other_tt]:
            ret.append(tt.trans_id_tree_path(''))
        return tuple(ret)

    def get_cset_path(self, parent, name):
        if name is None:
            if parent is not None:
                raise AssertionError()
            return None
        return pathjoin(self.cset.entries[parent].path, name)

    def add_file(self, parent, name, contents, executable, this=True, base=True, other=True, file_id=None):
        ret = []
        for i, (option, tt) in enumerate(self.selected_transforms(this, base, other)):
            if option is True:
                trans_id = tt.new_file(name, parent[i], [contents], executable=executable, file_id=file_id)
            else:
                trans_id = None
            ret.append(trans_id)
        return tuple(ret)

    def merge(self, merge_type=Merge3Merger, interesting_files=None, **kwargs):
        merger = self.make_merger(merge_type, interesting_files, **kwargs)
        merger.do_merge()
        return merger.cooked_conflicts

    def make_preview_transform(self):
        merger = self.make_merger(Merge3Merger, None, this_revision_tree=True)
        return merger.make_preview_transform()

    def make_merger(self, merge_type, interesting_files, this_revision_tree=False, **kwargs):
        self.base_tt.apply()
        self.base.commit('base commit')
        for tt, wt in ((self.this_tt, self.this), (self.other_tt, self.other)):
            wt.branch.pull(self.base.branch)
            wt.set_parent_ids([wt.branch.last_revision()])
            wt.flush()
            tt.apply()
            wt.commit('branch commit')
            wt.flush()
            if wt.branch.last_revision_info()[0] != 2:
                raise AssertionError()
        self.this.branch.fetch(self.other.branch)
        other_basis = self.other.branch.basis_tree()
        if this_revision_tree:
            self.this.commit('message')
            this_tree = self.this.basis_tree()
        else:
            this_tree = self.this
        merger = merge_type(this_tree, self.this, self.base, other_basis, interesting_files=interesting_files, do_merge=False, this_branch=self.this.branch, **kwargs)
        return merger

    def list_transforms(self):
        return [self.this_tt, self.base_tt, self.other_tt]

    def selected_transforms(self, this, base, other):
        pairs = [(this, self.this_tt), (base, self.base_tt), (other, self.other_tt)]
        return [(v, tt) for v, tt in pairs if v is not None]

    def add_symlink(self, parent, name, contents, file_id=None):
        ret = []
        for i, tt in enumerate(self.list_transforms()):
            trans_id = tt.new_symlink(name, parent[i], contents, file_id=file_id)
            ret.append(trans_id)
        return ret

    def remove_file(self, trans_ids, base=False, this=False, other=False):
        for trans_id, (option, tt) in zip(trans_ids, self.selected_transforms(this, base, other)):
            if option is True:
                tt.cancel_creation(trans_id)
                tt.cancel_versioning(trans_id)
                tt.set_executability(None, trans_id)

    def add_dir(self, parent, name, this=True, base=True, other=True, file_id=None):
        ret = []
        for i, (option, tt) in enumerate(self.selected_transforms(this, base, other)):
            if option is True:
                trans_id = tt.new_directory(name, parent[i], file_id)
            else:
                trans_id = None
            ret.append(trans_id)
        return tuple(ret)

    def change_name(self, trans_ids, base=None, this=None, other=None):
        for val, tt, trans_id in ((base, self.base_tt, trans_ids[0]), (this, self.this_tt, trans_ids[1]), (other, self.other_tt, trans_ids[2])):
            if val is None:
                continue
            parent_id = tt.final_parent(trans_id)
            tt.adjust_path(val, parent_id, trans_id)

    def change_parent(self, trans_ids, base=None, this=None, other=None):
        for trans_id, (parent, tt) in zip(trans_ids, self.selected_transforms(this, base, other)):
            parent_id = tt.trans_id_file_id(parent)
            tt.adjust_path(tt.final_name(trans_id), parent_id, trans_id)

    def change_contents(self, trans_id, base=None, this=None, other=None):
        for trans_id, (contents, tt) in zip(trans_id, self.selected_transforms(this, base, other)):
            tt.cancel_creation(trans_id)
            tt.create_file([contents], trans_id)

    def change_target(self, trans_ids, base=None, this=None, other=None):
        for trans_id, (target, tt) in zip(trans_ids, self.selected_transforms(this, base, other)):
            tt.cancel_creation(trans_id)
            tt.create_symlink(target, trans_id)

    def change_perms(self, trans_ids, base=None, this=None, other=None):
        for trans_id, (executability, tt) in zip(trans_ids, self.selected_transforms(this, base, other)):
            tt.set_executability(None, trans_id)
            tt.set_executability(executability, trans_id)

    def apply_inv_change(self, inventory_change, orig_inventory):
        orig_inventory_by_path = {}
        for file_id, path in orig_inventory.items():
            orig_inventory_by_path[path] = file_id

        def parent_id(file_id):
            try:
                parent_dir = os.path.dirname(orig_inventory[file_id])
            except:
                print(file_id)
                raise
            if parent_dir == '':
                return None
            return orig_inventory_by_path[parent_dir]

        def new_path(file_id):
            if fild_id in inventory_change:
                return inventory_change[file_id]
            else:
                parent = parent_id(file_id)
                if parent is None:
                    return orig_inventory[file_id]
                dirname = new_path(parent)
                return pathjoin(dirname, os.path.basename(orig_inventory[file_id]))
        new_inventory = {}
        for file_id in orig_inventory:
            path = new_path(file_id)
            if path is None:
                continue
            new_inventory[file_id] = path
        for file_id, path in inventory_change.items():
            if file_id in orig_inventory:
                continue
            new_inventory[file_id] = path
        return new_inventory

    def unlock(self):
        self.base.unlock()
        self.this.unlock()
        self.other.unlock()

    def cleanup(self):
        self.unlock()
        osutils.rmtree(self.dir)