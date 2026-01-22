import collections
import sys
import warnings
from . import protocols
from . import transports
from .log import logger
def _is_sslproto_available():
    return hasattr(ssl, 'MemoryBIO')