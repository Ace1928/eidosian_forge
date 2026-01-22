import inspect
import re
import typing as T
from collections import OrderedDict, namedtuple
from enum import IntEnum
from .common import (
class SectionType(IntEnum):
    """Types of sections."""
    SINGULAR = 0
    'For sections like examples.'
    MULTIPLE = 1
    'For sections like params.'
    SINGULAR_OR_MULTIPLE = 2
    'For sections like returns or yields.'