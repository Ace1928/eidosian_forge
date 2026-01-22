import re
import binascii
import email.quoprimime
import email.base64mime
from email.errors import HeaderParseError
from email import charset as _charset
def _ascii_split(self, fws, string, splitchars):
    parts = re.split('([' + FWS + ']+)', fws + string)
    if parts[0]:
        parts[:0] = ['']
    else:
        parts.pop(0)
    for fws, part in zip(*[iter(parts)] * 2):
        self._append_chunk(fws, part)