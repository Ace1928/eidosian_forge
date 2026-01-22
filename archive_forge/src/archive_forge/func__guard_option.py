import re
import zlib
import base64
from types import MappingProxyType
from numba.core import utils
def _guard_option(self, name):
    if name not in self.options:
        msg = f'{name!r} is not a valid option for {type(self)}'
        raise ValueError(msg)