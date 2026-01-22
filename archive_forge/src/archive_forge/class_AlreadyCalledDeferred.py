from . import version
import collections
from functools import wraps
import sys
import warnings
class AlreadyCalledDeferred(Exception):
    """The Deferred is already running a callback."""