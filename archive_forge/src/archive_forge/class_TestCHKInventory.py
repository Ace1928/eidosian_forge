from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
class TestCHKInventory(tests.TestCaseWithMemoryTransport):

    def get_chk_bytes(self):
        factory = groupcompress.make_pack_factory(True, True, 1)
        trans = self.get_transport('')
        return factory(trans)

    def read_bytes(self, chk_bytes, key):
        stream = chk_bytes.get_record_stream([key], 'unordered', True)
        return next(stream).get_bytes_as('fulltext')

    def test_deserialise_gives_CHKInventory(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        lines = chk_inv.to_lines()
        new_inv = CHKInventory.deserialise(chk_bytes, lines, (b'revid',))
        self.assertEqual(b'revid', new_inv.revision_id)
        self.assertEqual('directory', new_inv.root.kind)
        self.assertEqual(inv.root.file_id, new_inv.root.file_id)
        self.assertEqual(inv.root.parent_id, new_inv.root.parent_id)
        self.assertEqual(inv.root.name, new_inv.root.name)
        self.assertEqual(b'rootrev', new_inv.root.revision)
        self.assertEqual(b'plain', new_inv._search_key_name)

    def test_deserialise_wrong_revid(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        lines = chk_inv.to_lines()
        self.assertRaises(ValueError, CHKInventory.deserialise, chk_bytes, lines, (b'revid2',))

    def test_captures_rev_root_byid(self):
        inv = Inventory()
        inv.revision_id = b'foo'
        inv.root.revision = b'bar'
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        lines = chk_inv.to_lines()
        self.assertEqual([b'chkinventory:\n', b'revision_id: foo\n', b'root_id: TREE_ROOT\n', b'parent_id_basename_to_file_id: sha1:eb23f0ad4b07f48e88c76d4c94292be57fb2785f\n', b'id_to_entry: sha1:debfe920f1f10e7929260f0534ac9a24d7aabbb4\n'], lines)
        chk_inv = CHKInventory.deserialise(chk_bytes, lines, (b'foo',))
        self.assertEqual(b'plain', chk_inv._search_key_name)

    def test_captures_parent_id_basename_index(self):
        inv = Inventory()
        inv.revision_id = b'foo'
        inv.root.revision = b'bar'
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        lines = chk_inv.to_lines()
        self.assertEqual([b'chkinventory:\n', b'revision_id: foo\n', b'root_id: TREE_ROOT\n', b'parent_id_basename_to_file_id: sha1:eb23f0ad4b07f48e88c76d4c94292be57fb2785f\n', b'id_to_entry: sha1:debfe920f1f10e7929260f0534ac9a24d7aabbb4\n'], lines)
        chk_inv = CHKInventory.deserialise(chk_bytes, lines, (b'foo',))
        self.assertEqual(b'plain', chk_inv._search_key_name)

    def test_captures_search_key_name(self):
        inv = Inventory()
        inv.revision_id = b'foo'
        inv.root.revision = b'bar'
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv, search_key_name=b'hash-16-way')
        lines = chk_inv.to_lines()
        self.assertEqual([b'chkinventory:\n', b'search_key_name: hash-16-way\n', b'root_id: TREE_ROOT\n', b'parent_id_basename_to_file_id: sha1:eb23f0ad4b07f48e88c76d4c94292be57fb2785f\n', b'revision_id: foo\n', b'id_to_entry: sha1:debfe920f1f10e7929260f0534ac9a24d7aabbb4\n'], lines)
        chk_inv = CHKInventory.deserialise(chk_bytes, lines, (b'foo',))
        self.assertEqual(b'hash-16-way', chk_inv._search_key_name)

    def test_directory_children_on_demand(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        inv.add(InventoryFile(b'fileid', 'file', inv.root.file_id))
        inv.get_entry(b'fileid').revision = b'filerev'
        inv.get_entry(b'fileid').executable = True
        inv.get_entry(b'fileid').text_sha1 = b'ffff'
        inv.get_entry(b'fileid').text_size = 1
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        lines = chk_inv.to_lines()
        new_inv = CHKInventory.deserialise(chk_bytes, lines, (b'revid',))
        root_entry = new_inv.get_entry(inv.root.file_id)
        self.assertEqual(None, root_entry._children)
        self.assertEqual({'file'}, set(root_entry.children))
        file_direct = new_inv.get_entry(b'fileid')
        file_found = root_entry.children['file']
        self.assertEqual(file_direct.kind, file_found.kind)
        self.assertEqual(file_direct.file_id, file_found.file_id)
        self.assertEqual(file_direct.parent_id, file_found.parent_id)
        self.assertEqual(file_direct.name, file_found.name)
        self.assertEqual(file_direct.revision, file_found.revision)
        self.assertEqual(file_direct.text_sha1, file_found.text_sha1)
        self.assertEqual(file_direct.text_size, file_found.text_size)
        self.assertEqual(file_direct.executable, file_found.executable)

    def test_from_inventory_maximum_size(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv, 120)
        chk_inv.id_to_entry._ensure_root()
        self.assertEqual(120, chk_inv.id_to_entry._root_node.maximum_size)
        self.assertEqual(1, chk_inv.id_to_entry._root_node._key_width)
        p_id_basename = chk_inv.parent_id_basename_to_file_id
        p_id_basename._ensure_root()
        self.assertEqual(120, p_id_basename._root_node.maximum_size)
        self.assertEqual(2, p_id_basename._root_node._key_width)

    def test_iter_all_ids(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        inv.add(InventoryFile(b'fileid', 'file', inv.root.file_id))
        inv.get_entry(b'fileid').revision = b'filerev'
        inv.get_entry(b'fileid').executable = True
        inv.get_entry(b'fileid').text_sha1 = b'ffff'
        inv.get_entry(b'fileid').text_size = 1
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        lines = chk_inv.to_lines()
        new_inv = CHKInventory.deserialise(chk_bytes, lines, (b'revid',))
        fileids = sorted(new_inv.iter_all_ids())
        self.assertEqual([inv.root.file_id, b'fileid'], fileids)

    def test__len__(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        inv.add(InventoryFile(b'fileid', 'file', inv.root.file_id))
        inv.get_entry(b'fileid').revision = b'filerev'
        inv.get_entry(b'fileid').executable = True
        inv.get_entry(b'fileid').text_sha1 = b'ffff'
        inv.get_entry(b'fileid').text_size = 1
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        self.assertEqual(2, len(chk_inv))

    def test_get_entry(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        inv.add(InventoryFile(b'fileid', 'file', inv.root.file_id))
        inv.get_entry(b'fileid').revision = b'filerev'
        inv.get_entry(b'fileid').executable = True
        inv.get_entry(b'fileid').text_sha1 = b'ffff'
        inv.get_entry(b'fileid').text_size = 1
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        lines = chk_inv.to_lines()
        new_inv = CHKInventory.deserialise(chk_bytes, lines, (b'revid',))
        root_entry = new_inv.get_entry(inv.root.file_id)
        file_entry = new_inv.get_entry(b'fileid')
        self.assertEqual('directory', root_entry.kind)
        self.assertEqual(inv.root.file_id, root_entry.file_id)
        self.assertEqual(inv.root.parent_id, root_entry.parent_id)
        self.assertEqual(inv.root.name, root_entry.name)
        self.assertEqual(b'rootrev', root_entry.revision)
        self.assertEqual('file', file_entry.kind)
        self.assertEqual(b'fileid', file_entry.file_id)
        self.assertEqual(inv.root.file_id, file_entry.parent_id)
        self.assertEqual('file', file_entry.name)
        self.assertEqual(b'filerev', file_entry.revision)
        self.assertEqual(b'ffff', file_entry.text_sha1)
        self.assertEqual(1, file_entry.text_size)
        self.assertEqual(True, file_entry.executable)
        self.assertRaises(errors.NoSuchId, new_inv.get_entry, 'missing')

    def test_has_id_true(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        inv.add(InventoryFile(b'fileid', 'file', inv.root.file_id))
        inv.get_entry(b'fileid').revision = b'filerev'
        inv.get_entry(b'fileid').executable = True
        inv.get_entry(b'fileid').text_sha1 = b'ffff'
        inv.get_entry(b'fileid').text_size = 1
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        self.assertTrue(chk_inv.has_id(b'fileid'))
        self.assertTrue(chk_inv.has_id(inv.root.file_id))

    def test_has_id_not(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        self.assertFalse(chk_inv.has_id(b'fileid'))

    def test_id2path(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        direntry = InventoryDirectory(b'dirid', 'dir', inv.root.file_id)
        fileentry = InventoryFile(b'fileid', 'file', b'dirid')
        inv.add(direntry)
        inv.add(fileentry)
        inv.get_entry(b'fileid').revision = b'filerev'
        inv.get_entry(b'fileid').executable = True
        inv.get_entry(b'fileid').text_sha1 = b'ffff'
        inv.get_entry(b'fileid').text_size = 1
        inv.get_entry(b'dirid').revision = b'filerev'
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        lines = chk_inv.to_lines()
        new_inv = CHKInventory.deserialise(chk_bytes, lines, (b'revid',))
        self.assertEqual('', new_inv.id2path(inv.root.file_id))
        self.assertEqual('dir', new_inv.id2path(b'dirid'))
        self.assertEqual('dir/file', new_inv.id2path(b'fileid'))

    def test_path2id(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        direntry = InventoryDirectory(b'dirid', 'dir', inv.root.file_id)
        fileentry = InventoryFile(b'fileid', 'file', b'dirid')
        inv.add(direntry)
        inv.add(fileentry)
        inv.get_entry(b'fileid').revision = b'filerev'
        inv.get_entry(b'fileid').executable = True
        inv.get_entry(b'fileid').text_sha1 = b'ffff'
        inv.get_entry(b'fileid').text_size = 1
        inv.get_entry(b'dirid').revision = b'filerev'
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        lines = chk_inv.to_lines()
        new_inv = CHKInventory.deserialise(chk_bytes, lines, (b'revid',))
        self.assertEqual(inv.root.file_id, new_inv.path2id(''))
        self.assertEqual(b'dirid', new_inv.path2id('dir'))
        self.assertEqual(b'fileid', new_inv.path2id('dir/file'))

    def test_create_by_apply_delta_sets_root(self):
        inv = Inventory()
        inv.root.revision = b'myrootrev'
        inv.revision_id = b'revid'
        chk_bytes = self.get_chk_bytes()
        base_inv = CHKInventory.from_inventory(chk_bytes, inv)
        inv.add_path('', 'directory', b'myrootid', None)
        inv.revision_id = b'expectedid'
        inv.root.revision = b'myrootrev'
        reference_inv = CHKInventory.from_inventory(chk_bytes, inv)
        delta = [('', None, base_inv.root.file_id, None), (None, '', b'myrootid', inv.root)]
        new_inv = base_inv.create_by_apply_delta(delta, b'expectedid')
        self.assertEqual(reference_inv.root, new_inv.root)

    def test_create_by_apply_delta_empty_add_child(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        chk_bytes = self.get_chk_bytes()
        base_inv = CHKInventory.from_inventory(chk_bytes, inv)
        a_entry = InventoryFile(b'A-id', 'A', inv.root.file_id)
        a_entry.revision = b'filerev'
        a_entry.executable = True
        a_entry.text_sha1 = b'ffff'
        a_entry.text_size = 1
        inv.add(a_entry)
        inv.revision_id = b'expectedid'
        reference_inv = CHKInventory.from_inventory(chk_bytes, inv)
        delta = [(None, 'A', b'A-id', a_entry)]
        new_inv = base_inv.create_by_apply_delta(delta, b'expectedid')
        self.assertEqual(reference_inv.revision_id, new_inv.revision_id)
        self.assertEqual(reference_inv.root_id, new_inv.root_id)
        reference_inv.id_to_entry._ensure_root()
        new_inv.id_to_entry._ensure_root()
        self.assertEqual(reference_inv.id_to_entry._root_node._key, new_inv.id_to_entry._root_node._key)

    def test_create_by_apply_delta_empty_add_child_updates_parent_id(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        chk_bytes = self.get_chk_bytes()
        base_inv = CHKInventory.from_inventory(chk_bytes, inv)
        a_entry = InventoryFile(b'A-id', 'A', inv.root.file_id)
        a_entry.revision = b'filerev'
        a_entry.executable = True
        a_entry.text_sha1 = b'ffff'
        a_entry.text_size = 1
        inv.add(a_entry)
        inv.revision_id = b'expectedid'
        reference_inv = CHKInventory.from_inventory(chk_bytes, inv)
        delta = [(None, 'A', b'A-id', a_entry)]
        new_inv = base_inv.create_by_apply_delta(delta, b'expectedid')
        reference_inv.id_to_entry._ensure_root()
        reference_inv.parent_id_basename_to_file_id._ensure_root()
        new_inv.id_to_entry._ensure_root()
        new_inv.parent_id_basename_to_file_id._ensure_root()
        self.assertEqual(reference_inv.revision_id, new_inv.revision_id)
        self.assertEqual(reference_inv.root_id, new_inv.root_id)
        self.assertEqual(reference_inv.id_to_entry._root_node._key, new_inv.id_to_entry._root_node._key)
        self.assertEqual(reference_inv.parent_id_basename_to_file_id._root_node._key, new_inv.parent_id_basename_to_file_id._root_node._key)

    def test_iter_changes(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        inv.add(InventoryFile(b'fileid', 'file', inv.root.file_id))
        inv.get_entry(b'fileid').revision = b'filerev'
        inv.get_entry(b'fileid').executable = True
        inv.get_entry(b'fileid').text_sha1 = b'ffff'
        inv.get_entry(b'fileid').text_size = 1
        inv2 = Inventory()
        inv2.revision_id = b'revid2'
        inv2.root.revision = b'rootrev'
        inv2.add(InventoryFile(b'fileid', 'file', inv.root.file_id))
        inv2.get_entry(b'fileid').revision = b'filerev2'
        inv2.get_entry(b'fileid').executable = False
        inv2.get_entry(b'fileid').text_sha1 = b'bbbb'
        inv2.get_entry(b'fileid').text_size = 2
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        lines = chk_inv.to_lines()
        inv_1 = CHKInventory.deserialise(chk_bytes, lines, (b'revid',))
        chk_inv2 = CHKInventory.from_inventory(chk_bytes, inv2)
        lines = chk_inv2.to_lines()
        inv_2 = CHKInventory.deserialise(chk_bytes, lines, (b'revid2',))
        self.assertEqual([(b'fileid', ('file', 'file'), True, (True, True), (b'TREE_ROOT', b'TREE_ROOT'), ('file', 'file'), ('file', 'file'), (False, True))], list(inv_1.iter_changes(inv_2)))

    def test_parent_id_basename_to_file_id_index_enabled(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        inv.add(InventoryFile(b'fileid', 'file', inv.root.file_id))
        inv.get_entry(b'fileid').revision = b'filerev'
        inv.get_entry(b'fileid').executable = True
        inv.get_entry(b'fileid').text_sha1 = b'ffff'
        inv.get_entry(b'fileid').text_size = 1
        chk_bytes = self.get_chk_bytes()
        tmp_inv = CHKInventory.from_inventory(chk_bytes, inv)
        lines = tmp_inv.to_lines()
        chk_inv = CHKInventory.deserialise(chk_bytes, lines, (b'revid',))
        self.assertIsInstance(chk_inv.parent_id_basename_to_file_id, chk_map.CHKMap)
        self.assertEqual({(b'', b''): b'TREE_ROOT', (b'TREE_ROOT', b'file'): b'fileid'}, dict(chk_inv.parent_id_basename_to_file_id.iteritems()))

    def test_file_entry_to_bytes(self):
        inv = CHKInventory(None)
        ie = inventory.InventoryFile(b'file-id', 'filename', b'parent-id')
        ie.executable = True
        ie.revision = b'file-rev-id'
        ie.text_sha1 = b'abcdefgh'
        ie.text_size = 100
        bytes = inv._entry_to_bytes(ie)
        self.assertEqual(b'file: file-id\nparent-id\nfilename\nfile-rev-id\nabcdefgh\n100\nY', bytes)
        ie2 = inv._bytes_to_entry(bytes)
        self.assertEqual(ie, ie2)
        self.assertIsInstance(ie2.name, str)
        self.assertEqual((b'filename', b'file-id', b'file-rev-id'), inv._bytes_to_utf8name_key(bytes))

    def test_file2_entry_to_bytes(self):
        inv = CHKInventory(None)
        ie = inventory.InventoryFile(b'file-id', 'Ωname', b'parent-id')
        ie.executable = False
        ie.revision = b'file-rev-id'
        ie.text_sha1 = b'123456'
        ie.text_size = 25
        bytes = inv._entry_to_bytes(ie)
        self.assertEqual(b'file: file-id\nparent-id\n\xce\xa9name\nfile-rev-id\n123456\n25\nN', bytes)
        ie2 = inv._bytes_to_entry(bytes)
        self.assertEqual(ie, ie2)
        self.assertIsInstance(ie2.name, str)
        self.assertEqual((b'\xce\xa9name', b'file-id', b'file-rev-id'), inv._bytes_to_utf8name_key(bytes))

    def test_dir_entry_to_bytes(self):
        inv = CHKInventory(None)
        ie = inventory.InventoryDirectory(b'dir-id', 'dirname', b'parent-id')
        ie.revision = b'dir-rev-id'
        bytes = inv._entry_to_bytes(ie)
        self.assertEqual(b'dir: dir-id\nparent-id\ndirname\ndir-rev-id', bytes)
        ie2 = inv._bytes_to_entry(bytes)
        self.assertEqual(ie, ie2)
        self.assertIsInstance(ie2.name, str)
        self.assertEqual((b'dirname', b'dir-id', b'dir-rev-id'), inv._bytes_to_utf8name_key(bytes))

    def test_dir2_entry_to_bytes(self):
        inv = CHKInventory(None)
        ie = inventory.InventoryDirectory(b'dir-id', 'dirΩname', None)
        ie.revision = b'dir-rev-id'
        bytes = inv._entry_to_bytes(ie)
        self.assertEqual(b'dir: dir-id\n\ndir\xce\xa9name\ndir-rev-id', bytes)
        ie2 = inv._bytes_to_entry(bytes)
        self.assertEqual(ie, ie2)
        self.assertIsInstance(ie2.name, str)
        self.assertIs(ie2.parent_id, None)
        self.assertEqual((b'dir\xce\xa9name', b'dir-id', b'dir-rev-id'), inv._bytes_to_utf8name_key(bytes))

    def test_symlink_entry_to_bytes(self):
        inv = CHKInventory(None)
        ie = inventory.InventoryLink(b'link-id', 'linkname', b'parent-id')
        ie.revision = b'link-rev-id'
        ie.symlink_target = 'target/path'
        bytes = inv._entry_to_bytes(ie)
        self.assertEqual(b'symlink: link-id\nparent-id\nlinkname\nlink-rev-id\ntarget/path', bytes)
        ie2 = inv._bytes_to_entry(bytes)
        self.assertEqual(ie, ie2)
        self.assertIsInstance(ie2.name, str)
        self.assertIsInstance(ie2.symlink_target, str)
        self.assertEqual((b'linkname', b'link-id', b'link-rev-id'), inv._bytes_to_utf8name_key(bytes))

    def test_symlink2_entry_to_bytes(self):
        inv = CHKInventory(None)
        ie = inventory.InventoryLink(b'link-id', 'linkΩname', b'parent-id')
        ie.revision = b'link-rev-id'
        ie.symlink_target = 'target/Ωpath'
        bytes = inv._entry_to_bytes(ie)
        self.assertEqual(b'symlink: link-id\nparent-id\nlink\xce\xa9name\nlink-rev-id\ntarget/\xce\xa9path', bytes)
        ie2 = inv._bytes_to_entry(bytes)
        self.assertEqual(ie, ie2)
        self.assertIsInstance(ie2.name, str)
        self.assertIsInstance(ie2.symlink_target, str)
        self.assertEqual((b'link\xce\xa9name', b'link-id', b'link-rev-id'), inv._bytes_to_utf8name_key(bytes))

    def test_tree_reference_entry_to_bytes(self):
        inv = CHKInventory(None)
        ie = inventory.TreeReference(b'tree-root-id', 'treeΩname', b'parent-id')
        ie.revision = b'tree-rev-id'
        ie.reference_revision = b'ref-rev-id'
        bytes = inv._entry_to_bytes(ie)
        self.assertEqual(b'tree: tree-root-id\nparent-id\ntree\xce\xa9name\ntree-rev-id\nref-rev-id', bytes)
        ie2 = inv._bytes_to_entry(bytes)
        self.assertEqual(ie, ie2)
        self.assertIsInstance(ie2.name, str)
        self.assertEqual((b'tree\xce\xa9name', b'tree-root-id', b'tree-rev-id'), inv._bytes_to_utf8name_key(bytes))

    def make_basic_utf8_inventory(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        root_id = inv.root.file_id
        inv.add(InventoryFile(b'fileid', 'fïle', root_id))
        inv.get_entry(b'fileid').revision = b'filerev'
        inv.get_entry(b'fileid').text_sha1 = b'ffff'
        inv.get_entry(b'fileid').text_size = 0
        inv.add(InventoryDirectory(b'dirid', 'dir-€', root_id))
        inv.get_entry(b'dirid').revision = b'dirrev'
        inv.add(InventoryFile(b'childid', 'chïld', b'dirid'))
        inv.get_entry(b'childid').revision = b'filerev'
        inv.get_entry(b'childid').text_sha1 = b'ffff'
        inv.get_entry(b'childid').text_size = 0
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        lines = chk_inv.to_lines()
        return CHKInventory.deserialise(chk_bytes, lines, (b'revid',))

    def test__preload_handles_utf8(self):
        new_inv = self.make_basic_utf8_inventory()
        self.assertEqual({}, new_inv._fileid_to_entry_cache)
        self.assertFalse(new_inv._fully_cached)
        new_inv._preload_cache()
        self.assertEqual(sorted([new_inv.root_id, b'fileid', b'dirid', b'childid']), sorted(new_inv._fileid_to_entry_cache.keys()))
        ie_root = new_inv._fileid_to_entry_cache[new_inv.root_id]
        self.assertEqual(['dir-€', 'fïle'], sorted(ie_root._children.keys()))
        ie_dir = new_inv._fileid_to_entry_cache[b'dirid']
        self.assertEqual(['chïld'], sorted(ie_dir._children.keys()))

    def test__preload_populates_cache(self):
        inv = Inventory()
        inv.revision_id = b'revid'
        inv.root.revision = b'rootrev'
        root_id = inv.root.file_id
        inv.add(InventoryFile(b'fileid', 'file', root_id))
        inv.get_entry(b'fileid').revision = b'filerev'
        inv.get_entry(b'fileid').executable = True
        inv.get_entry(b'fileid').text_sha1 = b'ffff'
        inv.get_entry(b'fileid').text_size = 1
        inv.add(InventoryDirectory(b'dirid', 'dir', root_id))
        inv.get_entry(b'dirid').revision = b'dirrev'
        inv.add(InventoryFile(b'childid', 'child', b'dirid'))
        inv.get_entry(b'childid').revision = b'filerev'
        inv.get_entry(b'childid').executable = False
        inv.get_entry(b'childid').text_sha1 = b'dddd'
        inv.get_entry(b'childid').text_size = 1
        chk_bytes = self.get_chk_bytes()
        chk_inv = CHKInventory.from_inventory(chk_bytes, inv)
        lines = chk_inv.to_lines()
        new_inv = CHKInventory.deserialise(chk_bytes, lines, (b'revid',))
        self.assertEqual({}, new_inv._fileid_to_entry_cache)
        self.assertFalse(new_inv._fully_cached)
        new_inv._preload_cache()
        self.assertEqual(sorted([root_id, b'fileid', b'dirid', b'childid']), sorted(new_inv._fileid_to_entry_cache.keys()))
        self.assertTrue(new_inv._fully_cached)
        ie_root = new_inv._fileid_to_entry_cache[root_id]
        self.assertEqual(['dir', 'file'], sorted(ie_root._children.keys()))
        ie_dir = new_inv._fileid_to_entry_cache[b'dirid']
        self.assertEqual(['child'], sorted(ie_dir._children.keys()))

    def test__preload_handles_partially_evaluated_inventory(self):
        new_inv = self.make_basic_utf8_inventory()
        ie = new_inv.get_entry(new_inv.root_id)
        self.assertIs(None, ie._children)
        self.assertEqual(['dir-€', 'fïle'], sorted(ie.children.keys()))
        self.assertEqual(['dir-€', 'fïle'], sorted(ie._children.keys()))
        new_inv._preload_cache()
        self.assertEqual(['dir-€', 'fïle'], sorted(ie._children.keys()))
        ie_dir = new_inv.get_entry(b'dirid')
        self.assertEqual(['chïld'], sorted(ie_dir._children.keys()))

    def test_filter_change_in_renamed_subfolder(self):
        inv = Inventory(b'tree-root')
        inv.root.revision = b'rootrev'
        src_ie = inv.add_path('src', 'directory', b'src-id')
        src_ie.revision = b'srcrev'
        sub_ie = inv.add_path('src/sub/', 'directory', b'sub-id')
        sub_ie.revision = b'subrev'
        a_ie = inv.add_path('src/sub/a', 'file', b'a-id')
        a_ie.revision = b'filerev'
        a_ie.text_sha1 = osutils.sha_string(b'content\n')
        a_ie.text_size = len(b'content\n')
        chk_bytes = self.get_chk_bytes()
        inv = CHKInventory.from_inventory(chk_bytes, inv)
        inv = inv.create_by_apply_delta([('src/sub/a', 'src/sub/a', b'a-id', a_ie), ('src', 'src2', b'src-id', src_ie)], b'new-rev-2')
        new_inv = inv.filter([b'a-id', b'src-id'])
        self.assertEqual([('', b'tree-root'), ('src', b'src-id'), ('src/sub', b'sub-id'), ('src/sub/a', b'a-id')], [(path, ie.file_id) for path, ie in new_inv.iter_entries()])