import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def unescape_json_string(s: str) -> str:

    def unicode_escape_callback(match: Match[str]) -> str:
        return chr(int(match.group(1).upper(), 16))
    s = s.replace('\\"', '"').replace('\\b', '\x08').replace('\\r', '\r').replace('\\n', '\n').replace('\\t', '\t').replace('\\f', '\x0c').replace('\\/', '/').replace('\\\\', '\\')
    return re.sub('\\\\u([0-9A-Fa-f]{4})', unicode_escape_callback, s)