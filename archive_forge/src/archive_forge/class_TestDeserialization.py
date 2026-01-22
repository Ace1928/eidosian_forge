from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
class TestDeserialization(TestCase):
    """Test InventoryDeltaSerializer.parse_text_bytes."""

    def test_parse_no_bytes(self):
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, [])
        self.assertContainsRe(str(err), 'inventory delta is empty')

    def test_parse_bad_format(self):
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, [b'format: foo\n'])
        self.assertContainsRe(str(err), 'unknown format')

    def test_parse_no_parent(self):
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, [b'format: bzr inventory delta v1 (bzr 1.14)\n'])
        self.assertContainsRe(str(err), 'missing parent: marker')

    def test_parse_no_version(self):
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, [b'format: bzr inventory delta v1 (bzr 1.14)\n', b'parent: null:\n'])
        self.assertContainsRe(str(err), 'missing version: marker')

    def test_parse_duplicate_key_errors(self):
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        double_root_lines = b'format: bzr inventory delta v1 (bzr 1.14)\nparent: null:\nversion: null:\nversioned_root: true\ntree_references: true\nNone\x00/\x00an-id\x00\x00a@e\xc3\xa5ample.com--2004\x00dir\x00\x00\nNone\x00/\x00an-id\x00\x00a@e\xc3\xa5ample.com--2004\x00dir\x00\x00\n'
        err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(double_root_lines))
        self.assertContainsRe(str(err), 'duplicate file id')

    def test_parse_versioned_root_only(self):
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        parse_result = deserializer.parse_text_bytes(osutils.split_lines(root_only_lines))
        expected_entry = inventory.make_entry('directory', '', None, b'an-id')
        expected_entry.revision = b'a@e\xc3\xa5ample.com--2004'
        self.assertEqual((b'null:', b'entry-version', True, True, [(None, '', b'an-id', expected_entry)]), parse_result)

    def test_parse_special_revid_not_valid_last_mod(self):
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        root_only_lines = b'format: bzr inventory delta v1 (bzr 1.14)\nparent: null:\nversion: null:\nversioned_root: false\ntree_references: true\nNone\x00/\x00TREE_ROOT\x00\x00null:\x00dir\x00\x00\n'
        err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(root_only_lines))
        self.assertContainsRe(str(err), 'special revisionid found')

    def test_parse_versioned_root_versioned_disabled(self):
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        root_only_lines = b'format: bzr inventory delta v1 (bzr 1.14)\nparent: null:\nversion: null:\nversioned_root: false\ntree_references: true\nNone\x00/\x00TREE_ROOT\x00\x00a@e\xc3\xa5ample.com--2004\x00dir\x00\x00\n'
        err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(root_only_lines))
        self.assertContainsRe(str(err), 'Versioned root found')

    def test_parse_unique_root_id_root_versioned_disabled(self):
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        root_only_lines = b'format: bzr inventory delta v1 (bzr 1.14)\nparent: parent-id\nversion: a@e\xc3\xa5ample.com--2004\nversioned_root: false\ntree_references: true\nNone\x00/\x00an-id\x00\x00parent-id\x00dir\x00\x00\n'
        err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(root_only_lines))
        self.assertContainsRe(str(err), 'Versioned root found')

    def test_parse_unversioned_root_versioning_enabled(self):
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        parse_result = deserializer.parse_text_bytes(osutils.split_lines(root_only_unversioned))
        expected_entry = inventory.make_entry('directory', '', None, b'TREE_ROOT')
        expected_entry.revision = b'entry-version'
        self.assertEqual((b'null:', b'entry-version', False, False, [(None, '', b'TREE_ROOT', expected_entry)]), parse_result)

    def test_parse_versioned_root_when_disabled(self):
        deserializer = inventory_delta.InventoryDeltaDeserializer(allow_versioned_root=False)
        err = self.assertRaises(inventory_delta.IncompatibleInventoryDelta, deserializer.parse_text_bytes, osutils.split_lines(root_only_lines))
        self.assertEqual('versioned_root not allowed', str(err))

    def test_parse_tree_when_disabled(self):
        deserializer = inventory_delta.InventoryDeltaDeserializer(allow_tree_references=False)
        err = self.assertRaises(inventory_delta.IncompatibleInventoryDelta, deserializer.parse_text_bytes, osutils.split_lines(reference_lines))
        self.assertEqual('Tree reference not allowed', str(err))

    def test_parse_tree_when_header_disallows(self):
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        lines = b'format: bzr inventory delta v1 (bzr 1.14)\nparent: null:\nversion: entry-version\nversioned_root: false\ntree_references: false\nNone\x00/foo\x00id\x00TREE_ROOT\x00changed\x00tree\x00subtree-version\n'
        err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(lines))
        self.assertContainsRe(str(err), 'Tree reference found')

    def test_parse_versioned_root_when_header_disallows(self):
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        lines = b'format: bzr inventory delta v1 (bzr 1.14)\nparent: null:\nversion: entry-version\nversioned_root: false\ntree_references: false\nNone\x00/\x00TREE_ROOT\x00\x00a@e\xc3\xa5ample.com--2004\x00dir\n'
        err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(lines))
        self.assertContainsRe(str(err), 'Versioned root found')

    def test_parse_last_line_not_empty(self):
        """newpath must start with / if it is not None."""
        lines = root_only_lines[:-1]
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(lines))
        self.assertContainsRe(str(err), 'last line not empty')

    def test_parse_invalid_newpath(self):
        """newpath must start with / if it is not None."""
        lines = empty_lines
        lines += b'None\x00bad\x00TREE_ROOT\x00\x00version\x00dir\n'
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(lines))
        self.assertContainsRe(str(err), 'newpath invalid')

    def test_parse_invalid_oldpath(self):
        """oldpath must start with / if it is not None."""
        lines = root_only_lines
        lines += b'bad\x00/new\x00file-id\x00\x00version\x00dir\n'
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        err = self.assertRaises(InventoryDeltaError, deserializer.parse_text_bytes, osutils.split_lines(lines))
        self.assertContainsRe(str(err), 'oldpath invalid')

    def test_parse_new_file(self):
        """a new file is parsed correctly"""
        lines = root_only_lines
        fake_sha = b'deadbeef' * 5
        lines += b'None\x00/new\x00file-id\x00an-id\x00version\x00file\x00123\x00' + b'\x00' + fake_sha + b'\n'
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        parse_result = deserializer.parse_text_bytes(osutils.split_lines(lines))
        expected_entry = inventory.make_entry('file', 'new', b'an-id', b'file-id')
        expected_entry.revision = b'version'
        expected_entry.text_size = 123
        expected_entry.text_sha1 = fake_sha
        delta = parse_result[4]
        self.assertEqual((None, 'new', b'file-id', expected_entry), delta[-1])

    def test_parse_delete(self):
        lines = root_only_lines
        lines += b'/old-file\x00None\x00deleted-id\x00\x00null:\x00deleted\x00\x00\n'
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        parse_result = deserializer.parse_text_bytes(osutils.split_lines(lines))
        delta = parse_result[4]
        self.assertEqual(('old-file', None, b'deleted-id', None), delta[-1])