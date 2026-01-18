from array import array
import struct
import sys
import traceback
import types
from Xlib import X
from Xlib.support import lock
values, remdata = s.parse_binary(data, display, rawdict = 0)

        Convert a binary representation of the structure into Python values.

        DATA is a string or a buffer containing the binary data.
        DISPLAY should be a Xlib.protocol.display.Display object if
        there are any Resource fields or Lists with ResourceObjs.

        The Python values are returned as VALUES.  If RAWDICT is true,
        a Python dictionary is returned, where the keys are field
        names and the values are the corresponding Python value.  If
        RAWDICT is false, a DictWrapper will be returned where all
        fields are available as attributes.

        REMDATA are the remaining binary data, unused by the Struct object.

        