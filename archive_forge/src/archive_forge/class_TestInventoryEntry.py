from ... import errors, osutils, repository, revision, tests, workingtree
from ...tests.scenarios import load_tests_apply_scenarios
from .. import chk_map, groupcompress, inventory
from ..inventory import (ROOT_ID, CHKInventory, DuplicateFileId,
from . import TestCase, TestCaseWithTransport
class TestInventoryEntry(TestCase):

    def test_file_invalid_entry_name(self):
        self.assertRaises(InvalidEntryName, inventory.InventoryFile, b'123', 'a/hello.c', ROOT_ID)

    def test_file_backslash(self):
        file = inventory.InventoryFile(b'123', 'h\\ello.c', ROOT_ID)
        self.assertEqual(file.name, 'h\\ello.c')

    def test_file_kind_character(self):
        file = inventory.InventoryFile(b'123', 'hello.c', ROOT_ID)
        self.assertEqual(file.kind_character(), '')

    def test_dir_kind_character(self):
        dir = inventory.InventoryDirectory(b'123', 'hello.c', ROOT_ID)
        self.assertEqual(dir.kind_character(), '/')

    def test_link_kind_character(self):
        dir = inventory.InventoryLink(b'123', 'hello.c', ROOT_ID)
        self.assertEqual(dir.kind_character(), '')

    def test_tree_ref_kind_character(self):
        dir = TreeReference(b'123', 'hello.c', ROOT_ID)
        self.assertEqual(dir.kind_character(), '+')

    def test_dir_detect_changes(self):
        left = inventory.InventoryDirectory(b'123', 'hello.c', ROOT_ID)
        right = inventory.InventoryDirectory(b'123', 'hello.c', ROOT_ID)
        self.assertEqual((False, False), left.detect_changes(right))
        self.assertEqual((False, False), right.detect_changes(left))

    def test_file_detect_changes(self):
        left = inventory.InventoryFile(b'123', 'hello.c', ROOT_ID)
        left.text_sha1 = 123
        right = inventory.InventoryFile(b'123', 'hello.c', ROOT_ID)
        right.text_sha1 = 123
        self.assertEqual((False, False), left.detect_changes(right))
        self.assertEqual((False, False), right.detect_changes(left))
        left.executable = True
        self.assertEqual((False, True), left.detect_changes(right))
        self.assertEqual((False, True), right.detect_changes(left))
        right.text_sha1 = 321
        self.assertEqual((True, True), left.detect_changes(right))
        self.assertEqual((True, True), right.detect_changes(left))

    def test_symlink_detect_changes(self):
        left = inventory.InventoryLink(b'123', 'hello.c', ROOT_ID)
        left.symlink_target = 'foo'
        right = inventory.InventoryLink(b'123', 'hello.c', ROOT_ID)
        right.symlink_target = 'foo'
        self.assertEqual((False, False), left.detect_changes(right))
        self.assertEqual((False, False), right.detect_changes(left))
        left.symlink_target = 'different'
        self.assertEqual((True, False), left.detect_changes(right))
        self.assertEqual((True, False), right.detect_changes(left))

    def test_file_has_text(self):
        file = inventory.InventoryFile(b'123', 'hello.c', ROOT_ID)
        self.assertTrue(file.has_text())

    def test_directory_has_text(self):
        dir = inventory.InventoryDirectory(b'123', 'hello.c', ROOT_ID)
        self.assertFalse(dir.has_text())

    def test_link_has_text(self):
        link = inventory.InventoryLink(b'123', 'hello.c', ROOT_ID)
        self.assertFalse(link.has_text())

    def test_make_entry(self):
        self.assertIsInstance(inventory.make_entry('file', 'name', ROOT_ID), inventory.InventoryFile)
        self.assertIsInstance(inventory.make_entry('symlink', 'name', ROOT_ID), inventory.InventoryLink)
        self.assertIsInstance(inventory.make_entry('directory', 'name', ROOT_ID), inventory.InventoryDirectory)

    def test_make_entry_non_normalized(self):
        orig_normalized_filename = osutils.normalized_filename
        try:
            osutils.normalized_filename = osutils._accessible_normalized_filename
            entry = inventory.make_entry('file', 'å', ROOT_ID)
            self.assertEqual('å', entry.name)
            self.assertIsInstance(entry, inventory.InventoryFile)
            osutils.normalized_filename = osutils._inaccessible_normalized_filename
            self.assertRaises(errors.InvalidNormalization, inventory.make_entry, 'file', 'å', ROOT_ID)
        finally:
            osutils.normalized_filename = orig_normalized_filename