from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
class TestSerialization(TestCase):
    """Tests for InventoryDeltaSerializer.delta_to_lines."""

    def test_empty_delta_to_lines(self):
        old_inv = Inventory(None)
        new_inv = Inventory(None)
        delta = new_inv._make_delta(old_inv)
        serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=True)
        self.assertEqual(BytesIO(empty_lines).readlines(), serializer.delta_to_lines(NULL_REVISION, NULL_REVISION, delta))

    def test_root_only_to_lines(self):
        old_inv = Inventory(None)
        new_inv = Inventory(None)
        root = new_inv.make_entry('directory', '', None, b'an-id')
        root.revision = b'a@e\xc3\xa5ample.com--2004'
        new_inv.add(root)
        delta = new_inv._make_delta(old_inv)
        serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=True)
        self.assertEqual(BytesIO(root_only_lines).readlines(), serializer.delta_to_lines(NULL_REVISION, b'entry-version', delta))

    def test_unversioned_root(self):
        old_inv = Inventory(None)
        new_inv = Inventory(None)
        root = new_inv.make_entry('directory', '', None, b'TREE_ROOT')
        root.revision = b'entry-version'
        new_inv.add(root)
        delta = new_inv._make_delta(old_inv)
        serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=False, tree_references=False)
        serialized_lines = serializer.delta_to_lines(NULL_REVISION, b'entry-version', delta)
        self.assertEqual(BytesIO(root_only_unversioned).readlines(), serialized_lines)
        deserializer = inventory_delta.InventoryDeltaDeserializer()
        self.assertEqual((NULL_REVISION, b'entry-version', False, False, delta), deserializer.parse_text_bytes(serialized_lines))

    def test_unversioned_non_root_errors(self):
        old_inv = Inventory(None)
        new_inv = Inventory(None)
        root = new_inv.make_entry('directory', '', None, b'TREE_ROOT')
        root.revision = b'a@e\xc3\xa5ample.com--2004'
        new_inv.add(root)
        non_root = new_inv.make_entry('directory', 'foo', root.file_id, b'id')
        new_inv.add(non_root)
        delta = new_inv._make_delta(old_inv)
        serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=True)
        err = self.assertRaises(InventoryDeltaError, serializer.delta_to_lines, NULL_REVISION, b'entry-version', delta)
        self.assertContainsRe(str(err), "^no version for fileid b?'id'$")

    def test_richroot_unversioned_root_errors(self):
        old_inv = Inventory(None)
        new_inv = Inventory(None)
        root = new_inv.make_entry('directory', '', None, b'TREE_ROOT')
        new_inv.add(root)
        delta = new_inv._make_delta(old_inv)
        serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=True)
        err = self.assertRaises(InventoryDeltaError, serializer.delta_to_lines, NULL_REVISION, b'entry-version', delta)
        self.assertContainsRe(str(err), "no version for fileid b?'TREE_ROOT'$")

    def test_nonrichroot_versioned_root_errors(self):
        old_inv = Inventory(None)
        new_inv = Inventory(None)
        root = new_inv.make_entry('directory', '', None, b'TREE_ROOT')
        root.revision = b'a@e\xc3\xa5ample.com--2004'
        new_inv.add(root)
        delta = new_inv._make_delta(old_inv)
        serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=False, tree_references=True)
        err = self.assertRaises(InventoryDeltaError, serializer.delta_to_lines, NULL_REVISION, b'entry-version', delta)
        self.assertContainsRe(str(err), "^Version present for / in b?'TREE_ROOT'")

    def test_unknown_kind_errors(self):
        old_inv = Inventory(None)
        new_inv = Inventory(None)
        root = new_inv.make_entry('directory', '', None, b'my-rich-root-id')
        root.revision = b'changed'
        new_inv.add(root)

        class StrangeInventoryEntry(inventory.InventoryEntry):
            kind = 'strange'
        non_root = StrangeInventoryEntry(b'id', 'foo', root.file_id)
        non_root.revision = b'changed'
        new_inv.add(non_root)
        delta = new_inv._make_delta(old_inv)
        serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=True)
        err = self.assertRaises(KeyError, serializer.delta_to_lines, NULL_REVISION, b'entry-version', delta)
        self.assertEqual(('strange',), err.args)

    def test_tree_reference_disabled(self):
        old_inv = Inventory(None)
        new_inv = Inventory(None)
        root = new_inv.make_entry('directory', '', None, b'TREE_ROOT')
        root.revision = b'a@e\xc3\xa5ample.com--2004'
        new_inv.add(root)
        non_root = new_inv.make_entry('tree-reference', 'foo', root.file_id, b'id')
        non_root.revision = b'changed'
        non_root.reference_revision = b'subtree-version'
        new_inv.add(non_root)
        delta = new_inv._make_delta(old_inv)
        serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=False)
        err = self.assertRaises(KeyError, serializer.delta_to_lines, NULL_REVISION, b'entry-version', delta)
        self.assertEqual(('tree-reference',), err.args)

    def test_tree_reference_enabled(self):
        old_inv = Inventory(None)
        new_inv = Inventory(None)
        root = new_inv.make_entry('directory', '', None, b'TREE_ROOT')
        root.revision = b'a@e\xc3\xa5ample.com--2004'
        new_inv.add(root)
        non_root = new_inv.make_entry('tree-reference', 'foo', root.file_id, b'id')
        non_root.revision = b'changed'
        non_root.reference_revision = b'subtree-version'
        new_inv.add(non_root)
        delta = new_inv._make_delta(old_inv)
        serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=True)
        self.assertEqual(BytesIO(reference_lines).readlines(), serializer.delta_to_lines(NULL_REVISION, b'entry-version', delta))

    def test_to_inventory_root_id_versioned_not_permitted(self):
        root_entry = inventory.make_entry('directory', '', None, b'TREE_ROOT')
        root_entry.revision = b'some-version'
        delta = [(None, '', b'TREE_ROOT', root_entry)]
        serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=False, tree_references=True)
        self.assertRaises(InventoryDeltaError, serializer.delta_to_lines, b'old-version', b'new-version', delta)

    def test_to_inventory_root_id_not_versioned(self):
        delta = [(None, '', b'an-id', inventory.make_entry('directory', '', None, b'an-id'))]
        serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=True)
        self.assertRaises(InventoryDeltaError, serializer.delta_to_lines, b'old-version', b'new-version', delta)

    def test_to_inventory_has_tree_not_meant_to(self):
        make_entry = inventory.make_entry
        tree_ref = make_entry('tree-reference', 'foo', b'changed-in', b'ref-id')
        tree_ref.reference_revision = b'ref-revision'
        delta = [(None, '', b'an-id', make_entry('directory', '', b'changed-in', b'an-id')), (None, 'foo', b'ref-id', tree_ref)]
        serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=True)
        self.assertRaises(InventoryDeltaError, serializer.delta_to_lines, b'old-version', b'new-version', delta)

    def test_to_inventory_torture(self):

        def make_entry(kind, name, parent_id, file_id, **attrs):
            entry = inventory.make_entry(kind, name, parent_id, file_id)
            for name, value in attrs.items():
                setattr(entry, name, value)
            return entry
        delta = [(None, '', b'new-root-id', make_entry('directory', '', None, b'new-root-id', revision=b'changed-in')), ('', 'old-root', b'TREE_ROOT', make_entry('directory', 'subdir-now', b'new-root-id', b'TREE_ROOT', revision=b'moved-root')), ('under-old-root', 'old-root/under-old-root', b'moved-id', make_entry('file', 'under-old-root', b'TREE_ROOT', b'moved-id', revision=b'old-rev', executable=False, text_size=30, text_sha1=b'some-sha')), ('old-file', None, b'deleted-id', None), ('ref', 'ref', b'ref-id', make_entry('tree-reference', 'ref', b'new-root-id', b'ref-id', reference_revision=b'tree-reference-id', revision=b'new-rev')), ('dir/link', 'old-root/dir/link', b'link-id', make_entry('symlink', 'link', b'deep-id', b'link-id', symlink_target='target', revision=b'new-rev')), ('dir', 'old-root/dir', b'deep-id', make_entry('directory', 'dir', b'TREE_ROOT', b'deep-id', revision=b'new-rev')), (None, 'configure', b'exec-id', make_entry('file', 'configure', b'new-root-id', b'exec-id', executable=True, text_size=30, text_sha1=b'some-sha', revision=b'old-rev'))]
        serializer = inventory_delta.InventoryDeltaSerializer(versioned_root=True, tree_references=True)
        lines = serializer.delta_to_lines(NULL_REVISION, b'something', delta)
        expected = b'format: bzr inventory delta v1 (bzr 1.14)\nparent: null:\nversion: something\nversioned_root: true\ntree_references: true\n/\x00/old-root\x00TREE_ROOT\x00new-root-id\x00moved-root\x00dir\n/dir\x00/old-root/dir\x00deep-id\x00TREE_ROOT\x00new-rev\x00dir\n/dir/link\x00/old-root/dir/link\x00link-id\x00deep-id\x00new-rev\x00link\x00target\n/old-file\x00None\x00deleted-id\x00\x00null:\x00deleted\x00\x00\n/ref\x00/ref\x00ref-id\x00new-root-id\x00new-rev\x00tree\x00tree-reference-id\n/under-old-root\x00/old-root/under-old-root\x00moved-id\x00TREE_ROOT\x00old-rev\x00file\x0030\x00\x00some-sha\nNone\x00/\x00new-root-id\x00\x00changed-in\x00dir\nNone\x00/configure\x00exec-id\x00new-root-id\x00old-rev\x00file\x0030\x00Y\x00some-sha\n'
        serialized = b''.join(lines)
        self.assertIsInstance(serialized, bytes)
        self.assertEqual(expected, serialized)