import re
import base64
import binascii
import functools
from string import ascii_letters, digits
from email import errors
class _QByteMap(dict):
    safe = b'-!*+/' + ascii_letters.encode('ascii') + digits.encode('ascii')

    def __missing__(self, key):
        if key in self.safe:
            self[key] = chr(key)
        else:
            self[key] = '={:02X}'.format(key)
        return self[key]