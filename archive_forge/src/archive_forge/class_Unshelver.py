import errno
import re
import fastbencode as bencode
from . import errors
from .lazy_import import lazy_import
from breezy import (
from breezy.bzr import (
class Unshelver:
    """Unshelve shelved changes."""

    def __init__(self, tree, base_tree, transform, message):
        """Constructor.

        :param tree: The tree to apply the changes to.
        :param base_tree: The basis to apply the tranform to.
        :param message: A message from the shelved transform.
        """
        self.tree = tree
        self.base_tree = base_tree
        self.transform = transform
        self.message = message

    @staticmethod
    def iter_records(shelf_file):
        parser = pack.ContainerPushParser()
        parser.accept_bytes(shelf_file.read())
        return iter(parser.read_pending_records())

    @staticmethod
    def parse_metadata(records):
        names, metadata_bytes = next(records)
        if names[0] != (b'metadata',):
            raise ShelfCorrupt
        metadata = bencode.bdecode(metadata_bytes)
        message = metadata.get(b'message')
        if message is not None:
            metadata[b'message'] = message.decode('utf-8')
        return metadata

    @classmethod
    def from_tree_and_shelf(klass, tree, shelf_file):
        """Create an Unshelver from a tree and a shelf file.

        :param tree: The tree to apply shelved changes to.
        :param shelf_file: A file-like object containing shelved changes.
        :return: The Unshelver.
        """
        records = klass.iter_records(shelf_file)
        metadata = klass.parse_metadata(records)
        base_revision_id = metadata[b'revision_id']
        try:
            base_tree = tree.revision_tree(base_revision_id)
        except errors.NoSuchRevisionInTree:
            base_tree = tree.branch.repository.revision_tree(base_revision_id)
        tt = base_tree.preview_transform()
        tt.deserialize(records)
        return klass(tree, base_tree, tt, metadata.get(b'message'))

    def make_merger(self):
        """Return a merger that can unshelve the changes."""
        target_tree = self.transform.get_preview_tree()
        merger = merge.Merger.from_uncommitted(self.tree, target_tree, self.base_tree)
        merger.merge_type = merge.Merge3Merger
        return merger

    def finalize(self):
        """Release all resources held by this Unshelver."""
        self.transform.finalize()