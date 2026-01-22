from __future__ import print_function   # This version of olefile requires Python 2.7 or 3.5+.
import io
import sys
import struct, array, os.path, datetime, logging, warnings, traceback
def _decode_utf16_str(self, utf16_str, errors='replace'):
    """
        Decode a string encoded in UTF-16 LE format, as found in the OLE
        directory or in property streams. Return a string encoded
        according to the path_encoding specified for the OleFileIO object.

        :param bytes utf16_str: bytes string encoded in UTF-16 LE format
        :param str errors: str, see python documentation for str.decode()
        :return: str, encoded according to path_encoding
        :rtype: str
        """
    unicode_str = utf16_str.decode('UTF-16LE', errors)
    if self.path_encoding:
        return unicode_str.encode(self.path_encoding, errors)
    else:
        return unicode_str