import functools
import inspect
import reprlib
import sys
import traceback
from . import constants
Replacement for traceback.extract_stack() that only does the
    necessary work for asyncio debug mode.
    