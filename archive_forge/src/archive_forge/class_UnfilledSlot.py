from collections.abc import Sequence
from typing import Optional, Union, cast
from twisted.python.compat import nativeString
from twisted.web._responses import RESPONSES
class UnfilledSlot(Exception):
    """
    During flattening, a slot with no associated data was encountered.
    """