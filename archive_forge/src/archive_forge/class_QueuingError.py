import copy
from collections import OrderedDict
from contextlib import contextmanager
from threading import RLock
from typing import Optional
class QueuingError(Exception):
    """Exception that is raised when there is a queuing error"""