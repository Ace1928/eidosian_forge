import string
import tempfile
import time
from unittest import IsolatedAsyncioTestCase as TestCase
import aiosqlite
from .smoke import setup_logger

    Decorator for perf testing a block of async code.

    Expects the wrapped function to return an async generator.
    The generator should do setup, then yield when ready to start perf testing.
    The decorator will then pump the generator repeatedly until the target
    time has been reached, then close the generator and print perf results.
    