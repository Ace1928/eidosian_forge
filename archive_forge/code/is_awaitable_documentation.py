import inspect
from typing import Any
from types import CoroutineType, GeneratorType
Return true if object can be passed to an ``await`` expression.

    Instead of testing if the object is an instance of abc.Awaitable, it checks
    the existence of an `__await__` attribute. This is much faster.
    