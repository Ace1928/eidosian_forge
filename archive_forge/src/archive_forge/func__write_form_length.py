import struct
import builtins
import warnings
from collections import namedtuple
def _write_form_length(self, datalength):
    if self._aifc:
        commlength = 18 + 5 + len(self._compname)
        if commlength & 1:
            commlength = commlength + 1
        verslength = 12
    else:
        commlength = 18
        verslength = 0
    _write_ulong(self._file, 4 + verslength + self._marklength + 8 + commlength + 16 + datalength)
    return commlength