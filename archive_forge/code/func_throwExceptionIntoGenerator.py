import builtins
import copy
import inspect
import linecache
import sys
from inspect import getmro
from io import StringIO
from typing import Callable, NoReturn, TypeVar
import opcode
from twisted.python import reflect
@_extraneous
def throwExceptionIntoGenerator(self, g):
    """
        Throw the original exception into the given generator,
        preserving traceback information if available.

        @return: The next value yielded from the generator.
        @raise StopIteration: If there are no more values in the generator.
        @raise anything else: Anything that the generator raises.
        """
    return g.throw(self.value.with_traceback(self.tb))