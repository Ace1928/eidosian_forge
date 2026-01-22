from breezy import errors, osutils
from breezy.bzr import inventory
from breezy.bzr.inventory import (InventoryDirectory, InventoryEntry,
from breezy.bzr.tests.per_inventory import TestCaseWithInventory
class TestInventory(TestCaseWithInventory):

    def make_init_inventory(self):
        inv = inventory.Inventory(b'tree-root')
        inv.revision = b'initial-rev'
        inv.root.revision = b'initial-rev'
        return self.inv_to_test_inv(inv)

    def make_file(self, file_id, name, parent_id, content=b'content\n', revision=b'new-test-rev'):
        ie = InventoryFile(file_id, name, parent_id)
        ie.text_sha1 = osutils.sha_string(content)
        ie.text_size = len(content)
        ie.revision = revision
        return ie

    def make_link(self, file_id, name, parent_id, target='link-target\n'):
        ie = InventoryLink(file_id, name, parent_id)
        ie.symlink_target = target
        return ie

    def prepare_inv_with_nested_dirs(self):
        inv = inventory.Inventory(b'tree-root')
        inv.root.revision = b'revision'
        for args in [('src', 'directory', b'src-id'), ('doc', 'directory', b'doc-id'), ('src/hello.c', 'file', b'hello-id'), ('src/bye.c', 'file', b'bye-id'), ('zz', 'file', b'zz-id'), ('src/sub/', 'directory', b'sub-id'), ('src/zz.c', 'file', b'zzc-id'), ('src/sub/a', 'file', b'a-id'), ('Makefile', 'file', b'makefile-id')]:
            ie = inv.add_path(*args)
            ie.revision = b'revision'
            if args[1] == 'file':
                ie.text_sha1 = osutils.sha_string(b'content\n')
                ie.text_size = len(b'content\n')
        return self.inv_to_test_inv(inv)