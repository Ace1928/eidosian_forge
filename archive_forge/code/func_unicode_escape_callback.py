import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def unicode_escape_callback(match: Match[str]) -> str:
    return chr(int(match.group(1).upper(), 16))