from __future__ import absolute_import
from future.utils import PY2
from itertools import islice
from future.backports.misc import count   # with step parameter on Py2.6
def __getitem_slice(self, slce):
    """Return a range which represents the requested slce
        of the sequence represented by this range.
        """
    scaled_indices = (self._step * n for n in slce.indices(self._len))
    start_offset, stop_offset, new_step = scaled_indices
    return newrange(self._start + start_offset, self._start + stop_offset, new_step)