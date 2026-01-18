import string
import tempfile
import time
from unittest import IsolatedAsyncioTestCase as TestCase
import aiosqlite
from .smoke import setup_logger
def timed(fn, name=None):
    """
    Decorator for perf testing a block of async code.

    Expects the wrapped function to return an async generator.
    The generator should do setup, then yield when ready to start perf testing.
    The decorator will then pump the generator repeatedly until the target
    time has been reached, then close the generator and print perf results.
    """
    name = name or fn.__name__

    async def wrapper(*args, **kwargs):
        gen = fn(*args, **kwargs)
        await gen.asend(None)
        count = 0
        before = time.time()
        while True:
            count += 1
            value = time.time() - before < TARGET
            try:
                if value:
                    await gen.asend(value)
                else:
                    await gen.aclose()
                    break
            except StopAsyncIteration:
                break
            except Exception as e:
                print(f'exception occurred: {e}')
                return
        duration = time.time() - before
        RESULTS[name] = (count, duration)
    return wrapper