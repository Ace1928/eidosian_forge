from io import BytesIO
from ... import osutils
from ...revision import NULL_REVISION
from .. import inventory, inventory_delta
from ..inventory import Inventory
from ..inventory_delta import InventoryDeltaError
from . import TestCase
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