import re
import zlib
import base64
from types import MappingProxyType
from numba.core import utils
@classmethod
def top_or_none(cls):
    """Get the TOS or return None if no config is set.
        """
    self = cls()
    if self:
        flags = self.top()
    else:
        flags = None
    return flags