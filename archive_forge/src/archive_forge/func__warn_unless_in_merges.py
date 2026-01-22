from fastimport import helpers, processor
from ... import debug, errors, osutils, revision
from ...bzr import generate_ids, inventory, serializer
from ...trace import mutter, note, warning
from .helpers import mode_to_kind
def _warn_unless_in_merges(self, fileid, path):
    if len(self.parents) <= 1:
        return
    for parent in self.parents[1:]:
        if fileid in self.get_inventory(parent):
            return
    self.warning('ignoring delete of %s as not in parent inventories', path)