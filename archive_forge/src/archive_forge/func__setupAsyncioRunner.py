import asyncio
import contextvars
import inspect
import warnings
from .case import TestCase
def _setupAsyncioRunner(self):
    assert self._asyncioRunner is None, 'asyncio runner is already initialized'
    runner = asyncio.Runner(debug=True)
    self._asyncioRunner = runner