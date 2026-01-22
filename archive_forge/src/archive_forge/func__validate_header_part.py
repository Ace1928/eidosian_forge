import codecs
import contextlib
import io
import os
import re
import socket
import struct
import sys
import tempfile
import warnings
import zipfile
from collections import OrderedDict
from pip._vendor.urllib3.util import make_headers, parse_url
from . import certs
from .__version__ import __version__
from ._internal_utils import (  # noqa: F401
from .compat import (
from .compat import parse_http_list as _parse_list_header
from .compat import (
from .cookies import cookiejar_from_dict
from .exceptions import (
from .structures import CaseInsensitiveDict
def _validate_header_part(header, header_part, header_validator_index):
    if isinstance(header_part, str):
        validator = _HEADER_VALIDATORS_STR[header_validator_index]
    elif isinstance(header_part, bytes):
        validator = _HEADER_VALIDATORS_BYTE[header_validator_index]
    else:
        raise InvalidHeader(f'Header part ({header_part!r}) from {header} must be of type str or bytes, not {type(header_part)}')
    if not validator.match(header_part):
        header_kind = 'name' if header_validator_index == 0 else 'value'
        raise InvalidHeader(f'Invalid leading whitespace, reserved character(s), or returncharacter(s) in header {header_kind}: {header_part!r}')