import os
import zlib
import logging
from io import BytesIO
import numpy as np
from ..core import Format, read_n_bytes, image_as_uint
def _read_one_tag(self):
    """
            Return (True, loc, size, T, L1) if an image that we can read.
            Return (False, loc, size, T, L1) if any other tag.
            """
    head = self._fp_read(6)
    if not head:
        raise IndexError('Reached end of swf movie')
    T, L1, L2 = _swf.get_type_and_len(head)
    if not L2:
        raise RuntimeError('Invalid tag length, could not proceed')
    isimage = False
    sze = L2 - 6
    if T == 0:
        raise IndexError('Reached end of swf movie')
    elif T in [20, 36]:
        isimage = True
    elif T in [6, 21, 35, 90]:
        logger.warning('Ignoring JPEG image: cannot read JPEG.')
    else:
        pass
    return (isimage, sze, T, L1)