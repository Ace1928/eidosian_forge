import collections
from tensorflow.python.autograph.pyct import origin_info
from tensorflow.python.util import traceback_utils
class FrameInfo(collections.namedtuple('FrameInfo', ('filename', 'lineno', 'function_name', 'code', 'is_converted', 'is_allowlisted'))):
    __slots__ = ()