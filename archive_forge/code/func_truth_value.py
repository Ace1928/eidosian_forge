from __future__ import annotations
import re
import sys
import warnings
from typing import (
from urllib.parse import unquote_plus
from pymongo.client_options import _parse_ssl_options
from pymongo.common import (
from pymongo.errors import ConfigurationError, InvalidURI
from pymongo.srv_resolver import _HAVE_DNSPYTHON, _SrvResolver
from pymongo.typings import _Address
def truth_value(val: Any) -> Any:
    if val in ('true', 'false'):
        return val == 'true'
    if isinstance(val, bool):
        return val
    return val