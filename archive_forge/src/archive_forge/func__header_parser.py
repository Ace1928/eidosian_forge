import sys
import email.parser
from .encoder import encode_with
from requests.structures import CaseInsensitiveDict
def _header_parser(string, encoding):
    major = sys.version_info[0]
    if major == 3:
        string = string.decode(encoding)
    headers = email.parser.HeaderParser().parsestr(string).items()
    return ((encode_with(k, encoding), encode_with(v, encoding)) for k, v in headers)