import hashlib
from boto.glacier.utils import chunk_hashes, tree_hash, bytes_to_hex
from boto.glacier.utils import compute_hashes_from_fileobj
def _insert_tree_hash(self, index, raw_tree_hash):
    list_length = len(self._tree_hashes)
    if index >= list_length:
        self._tree_hashes.extend([None] * (list_length - index + 1))
    self._tree_hashes[index] = raw_tree_hash