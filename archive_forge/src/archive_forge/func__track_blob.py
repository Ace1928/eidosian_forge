from __future__ import absolute_import
from .. import (
from ..helpers import (
import stat
def _track_blob(self, mark):
    if mark in self.blob_ref_counts:
        self.blob_ref_counts[mark] += 1
        pass
    elif mark in self.blobs['used']:
        self.blob_ref_counts[mark] = 2
        self.blobs['used'].remove(mark)
    elif mark in self.blobs['new']:
        self.blobs['used'].add(mark)
        self.blobs['new'].remove(mark)
    else:
        self.blobs['unknown'].add(mark)