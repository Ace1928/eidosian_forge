import os
import sys
import threading
import time
import traceback
import warnings
import weakref
import builtins
import pickle
import numpy as np
from ..util import cprint
class ClosedError(Exception):
    """Raised when an event handler receives a request to close the connection
    or discovers that the connection has been closed."""
    pass