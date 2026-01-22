import sys
from typing import AnyStr, Iterable, Optional
from constantly import NamedConstant
from incremental import Version
from twisted.python.deprecate import deprecatedProperty
from ._levels import LogLevel
from ._logger import Logger

        Template for unsupported operations.

        @param args: Arguments.
        