import contextlib
from datetime import datetime
import sys
import time
def _skipped_summary(self):
    """Get a summary string for skipped data."""
    string_pieces = []
    if self._num_tensors_skipped:
        string_pieces.append('%d tensors (%s)' % (self._num_tensors_skipped, readable_bytes_string(self._tensor_bytes_skipped)))
    if self._num_blobs_skipped:
        string_pieces.append('%d binary objects (%s)' % (self._num_blobs_skipped, readable_bytes_string(self._blob_bytes_skipped)))
    return ', '.join(string_pieces)