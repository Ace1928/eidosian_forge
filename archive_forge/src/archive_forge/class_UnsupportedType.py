from collections.abc import Sequence
from typing import Optional, Union, cast
from twisted.python.compat import nativeString
from twisted.web._responses import RESPONSES
class UnsupportedType(Exception):
    """
    During flattening, an object of a type which cannot be flattened was
    encountered.
    """