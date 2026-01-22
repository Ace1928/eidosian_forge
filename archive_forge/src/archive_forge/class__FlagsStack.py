import re
import zlib
import base64
from types import MappingProxyType
from numba.core import utils
class _FlagsStack(utils.ThreadLocalStack, stack_name='flags'):
    pass