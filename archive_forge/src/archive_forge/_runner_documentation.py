from os import kill
from signal import SIGTERM
from sys import stderr
from typing import Any, Callable, Mapping, TextIO
from attr import Factory, attrib, attrs
from constantly import NamedConstant
from twisted.internet.interfaces import IReactorCore
from twisted.logger import (
from ._exit import ExitStatus, exit
from ._pidfile import AlreadyRunningError, InvalidPIDFileError, IPIDFile, nonePIDFile

        Call C{self._reactorExited} with C{self._reactorExitedArguments}.

        @note: This method is called after the reactor exits.
        