import sys
import warnings
from typing import IO, Any, Iterable, Optional, Type
from twisted.python.compat import currentframe
from twisted.python.reflect import qual
from ._buffer import LimitedHistoryLogObserver
from ._file import FileLogObserver
from ._filter import FilteringLogObserver, LogLevelFilterPredicate
from ._format import eventAsText
from ._interfaces import ILogObserver
from ._io import LoggingFile
from ._levels import LogLevel
from ._logger import Logger
from ._observer import LogPublisher

        Twisted-enabled wrapper around L{warnings.showwarning}.

        If C{file} is L{None}, the default behaviour is to emit the warning to
        the log system, otherwise the original L{warnings.showwarning} Python
        function is called.

        @param message: A warning message to emit.
        @param category: A warning category to associate with C{message}.
        @param filename: A file name for the source code file issuing the
            warning.
        @param lineno: A line number in the source file where the warning was
            issued.
        @param file: A file to write the warning message to.  If L{None},
            write to L{sys.stderr}.
        @param line: A line of source code to include with the warning message.
            If L{None}, attempt to read the line from C{filename} and
            C{lineno}.
        