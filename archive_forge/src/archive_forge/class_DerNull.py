import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
class DerNull(DerObject):
    """Class to model a DER NULL element."""

    def __init__(self):
        """Initialize the DER object as a NULL."""
        DerObject.__init__(self, 5, b'', None, False)