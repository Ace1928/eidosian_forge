import re
import math
from calendar import isleap, leapdays
from decimal import Decimal
from operator import attrgetter
from urllib.parse import urlsplit
from typing import Any, Iterator, List, Match, Optional, Union, SupportsFloat
def upper_camel_case(s: str) -> str:
    return re.sub('^\\d+', '', re.sub('[\\W_]', '', s.title()))