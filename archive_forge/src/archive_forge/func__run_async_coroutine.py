import asyncio  # Essential for asynchronous operations. Documentation: https://docs.python.org/3/library/asyncio.html
import collections  # Provides support for container datatypes. Documentation: https://docs.python.org/3/library/collections.html
import collections.abc  # Offers abstract base classes for collections. Documentation: https://docs.python.org/3/library/collections.abc.html
import functools  # Utilities for higher-order functions and operations on callable objects. Documentation: https://docs.python.org/3/library/functools.html
import importlib.util  # Facilitates dynamic module loading. Documentation: https://docs.python.org/3/library/importlib.html#importlib.util
import inspect  # Inspects live objects. Documentation: https://docs.python.org/3/library/inspect.html
import logging  # Facilitates logging capabilities. Documentation: https://docs.python.org/3/library/logging.html
import logging.handlers  # Additional handlers for logging. Documentation: https://docs.python.org/3/library/logging.handlers.html
import os  # Interaction with the operating system. Documentation: https://docs.python.org/3/library/os.html
import pickle  # Object serialization and deserialization. Documentation: https://docs.python.org/3/library/pickle.html
import psutil  # System monitoring and resource management. Documentation: https://psutil.readthedocs.io/en/latest/
import random  # Generates pseudo-random numbers. Documentation: https://docs.python.org/3/library/random.html
import signal  # Set handlers for asynchronous events. Documentation: https://docs.python.org/3/library/signal.html
import sys  # Access to some variables used or maintained by the Python interpreter. Documentation: https://docs.python.org/3/library/sys.html
import threading  # Higher-level threading interface. Documentation: https://docs.python.org/3/library/threading.html
import time  # Time access and conversions. Documentation: https://docs.python.org/3/library/time.html
import tracemalloc  # Trace memory allocations. Documentation: https://docs.python.org/3/library/tracemalloc.html
import types  # Dynamic creation of new types. Documentation: https://docs.python.org/3/library/types.html
from types import (
import aiofiles  # Asynchronous file operations. Documentation: https://aiofiles.readthedocs.io/en/latest/
from aiofile import (
import numpy as np  # Fundamental package for scientific computing. Documentation: https://numpy.org/doc/
from typing import (  # Typing constructs for type hinting. Documentation: https://docs.python.org/3/library/typing.html
from datetime import (
from functools import (
from inspect import (  # Inspection and introspection of live objects. Documentation: https://docs.python.org/3/library/inspect.html
from numpy import (  # Numerical operations and array processing. Documentation: https://numpy.org/doc/stable/reference/routines.math.html
from math import (  # Mathematical functions. Documentation: https://docs.python.org/3/library/math.html
def _run_async_coroutine(self, coroutine):
    """

        Adapts the execution of a coroutine based on the current event loop state, ensuring compatibility with both

        synchronous and asynchronous contexts.

        Args:

            coroutine: The coroutine to be executed.

        Returns:

            The result of the coroutine execution if the event loop is not running; otherwise, schedules the coroutine

            for future execution.

        """
    try:
        loop = asyncio.get_running_loop()
        if loop.is_running():
            return asyncio.ensure_future(coroutine, loop=loop)
    except RuntimeError:
        return asyncio.run(coroutine)