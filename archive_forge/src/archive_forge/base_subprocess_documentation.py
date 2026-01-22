import collections
import subprocess
import warnings
from . import protocols
from . import transports
from .log import logger
Wait until the process exit and return the process return code.

        This method is a coroutine.