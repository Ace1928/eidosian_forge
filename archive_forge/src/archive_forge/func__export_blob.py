import stat
from typing import Dict, Tuple
from fastimport import commands, parser, processor
from fastimport import errors as fastimport_errors
from .index import commit_tree
from .object_store import iter_tree_contents
from .objects import ZERO_SHA, Blob, Commit, Tag
def _export_blob(self, blob):
    marker = self._allocate_marker()
    self.markers[marker] = blob.id
    return (commands.BlobCommand(marker, blob.data), marker)