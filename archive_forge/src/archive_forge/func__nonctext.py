import re
import binascii
import email.quoprimime
import email.base64mime
from email.errors import HeaderParseError
from email import charset as _charset
def _nonctext(self, s):
    """True if string s is not a ctext character of RFC822.
        """
    return s.isspace() or s in ('(', ')', '\\')