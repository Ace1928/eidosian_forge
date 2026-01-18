import sys
import email.parser
from .encoder import encode_with
from requests.structures import CaseInsensitiveDict
def test_part(part):
    return part != b'' and part != b'\r\n' and (part[:4] != b'--\r\n') and (part != b'--')