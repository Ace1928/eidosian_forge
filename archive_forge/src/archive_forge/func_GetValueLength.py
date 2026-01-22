from __future__ import division
import os
import struct
from pyu2f import errors
from pyu2f.hid import base
def GetValueLength(rd, pos):
    """Get value length for a key in rd.

  For a key at position pos in the Report Descriptor rd, return the length
  of the associated value.  This supports both short and long format
  values.

  Args:
    rd: Report Descriptor
    pos: The position of the key in rd.

  Returns:
    (key_size, data_len) where key_size is the number of bytes occupied by
    the key and data_len is the length of the value associated by the key.
  """
    rd = bytearray(rd)
    key = rd[pos]
    if key == LONG_ITEM_ENCODING:
        if pos + 1 < len(rd):
            return (3, rd[pos + 1])
        else:
            raise errors.HidError('Malformed report descriptor')
    else:
        code = key & 3
        if code <= 2:
            return (1, code)
        elif code == 3:
            return (1, 4)
    raise errors.HidError('Cannot happen')