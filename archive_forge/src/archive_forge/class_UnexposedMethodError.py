from collections.abc import Sequence
from typing import Optional, Union, cast
from twisted.python.compat import nativeString
from twisted.web._responses import RESPONSES
class UnexposedMethodError(Exception):
    """
    Raised on any attempt to get a method which has not been exposed.
    """