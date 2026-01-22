import bz2
import os
import sys
import tempfile
from io import BytesIO
from ... import diff, errors, merge, osutils
from ... import revision as _mod_revision
from ... import tests
from ... import transport as _mod_transport
from ... import treebuilder
from ...tests import features, test_commit
from ...tree import InterTree
from .. import bzrdir, inventory, knitrepo
from ..bundle.apply_bundle import install_bundle, merge_bundle
from ..bundle.bundle_data import BundleTree
from ..bundle.serializer import read_bundle, v4, v09, write_bundle
from ..bundle.serializer.v4 import BundleSerializerV4
from ..bundle.serializer.v08 import BundleSerializerV08
from ..bundle.serializer.v09 import BundleSerializerV09
from ..inventorytree import InventoryTree
class MockTree(InventoryTree):

    def __init__(self):
        from ..inventory import ROOT_ID, InventoryDirectory
        object.__init__(self)
        self.paths = {ROOT_ID: ''}
        self.ids = {'': ROOT_ID}
        self.contents = {}
        self.root = InventoryDirectory(ROOT_ID, '', None)
    inventory = property(lambda x: x)
    root_inventory = property(lambda x: x)

    def get_root_id(self):
        return self.root.file_id

    def all_file_ids(self):
        return set(self.paths.keys())

    def all_versioned_paths(self):
        return set(self.paths.values())

    def is_executable(self, path):
        return False

    def __getitem__(self, file_id):
        if file_id == self.root.file_id:
            return self.root
        else:
            return self.make_entry(file_id, self.paths[file_id])

    def get_entry_by_path(self, path):
        return self[self.path2id(path)]

    def parent_id(self, file_id):
        parent_dir = os.path.dirname(self.paths[file_id])
        if parent_dir == '':
            return None
        return self.ids[parent_dir]

    def iter_entries(self):
        for path, file_id in self.ids.items():
            yield (path, self[file_id])

    def kind(self, path):
        if path in self.contents:
            kind = 'file'
        else:
            kind = 'directory'
        return kind

    def make_entry(self, file_id, path):
        from ..inventory import InventoryDirectory, InventoryFile, InventoryLink
        if not isinstance(file_id, bytes):
            raise TypeError(file_id)
        name = os.path.basename(path)
        kind = self.kind(path)
        parent_id = self.parent_id(file_id)
        text_sha_1, text_size = self.contents_stats(path)
        if kind == 'directory':
            ie = InventoryDirectory(file_id, name, parent_id)
        elif kind == 'file':
            ie = InventoryFile(file_id, name, parent_id)
            ie.text_sha1 = text_sha_1
            ie.text_size = text_size
        elif kind == 'symlink':
            ie = InventoryLink(file_id, name, parent_id)
        else:
            raise errors.BzrError('unknown kind %r' % kind)
        return ie

    def add_dir(self, file_id, path):
        if not isinstance(file_id, bytes):
            raise TypeError(file_id)
        self.paths[file_id] = path
        self.ids[path] = file_id

    def add_file(self, file_id, path, contents):
        if not isinstance(file_id, bytes):
            raise TypeError(file_id)
        self.add_dir(file_id, path)
        self.contents[path] = contents

    def path2id(self, path):
        return self.ids.get(path)

    def id2path(self, file_id, recurse='down'):
        try:
            return self.paths[file_id]
        except KeyError:
            raise errors.NoSuchId(file_id, self)

    def get_file(self, path):
        result = BytesIO()
        try:
            result.write(self.contents[path])
        except KeyError:
            raise _mod_transport.NoSuchFile(path)
        result.seek(0, 0)
        return result

    def get_file_revision(self, path):
        return self.inventory.get_entry_by_path(path).revision

    def get_file_size(self, path):
        return self.inventory.get_entry_by_path(path).text_size

    def get_file_sha1(self, path, file_id=None):
        return self.inventory.get_entry_by_path(path).text_sha1

    def contents_stats(self, path):
        if path not in self.contents:
            return (None, None)
        text_sha1 = osutils.sha_file(self.get_file(path))
        return (text_sha1, len(self.contents[path]))