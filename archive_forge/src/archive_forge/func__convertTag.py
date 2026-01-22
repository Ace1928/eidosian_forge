import struct
from Cryptodome.Util.py3compat import byte_string, bchr, bord
from Cryptodome.Util.number import long_to_bytes, bytes_to_long
def _convertTag(self, tag):
    """Check if *tag* is a real DER tag (5 bits).
                Convert it from a character to number if necessary.
                """
    if not _is_number(tag):
        if len(tag) == 1:
            tag = bord(tag[0])
    if not (_is_number(tag) and 0 <= tag < 31):
        raise ValueError('Wrong DER tag')
    return tag