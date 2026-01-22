import re
import binascii
import email.quoprimime
import email.base64mime
from email.errors import HeaderParseError
from email import charset as _charset
def _maxlengths(self):
    yield (self._maxlen - len(self._current_line))
    while True:
        yield (self._maxlen - self._continuation_ws_len)