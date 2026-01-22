from __future__ import annotations
from datetime import datetime as DateTime
from typing import Any, Callable, Iterator, Mapping, Optional, Union, cast
from constantly import NamedConstant
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.python.failure import Failure
from twisted.python.reflect import safe_repr
from ._flatten import aFormatter, flatFormat
from ._interfaces import LogEvent
def _formatSystem(event: LogEvent) -> str:
    """
    Format the system specified in the event in the "log_system" key if set,
    otherwise the C{"log_namespace"} and C{"log_level"}, joined by a C{"#"}.
    Each defaults to C{"-"} is not set.  If formatting fails completely,
    "UNFORMATTABLE" is returned.

    @param event: The event containing the system specification.

    @return: A formatted string representing the "log_system" key.
    """
    system = cast(Optional[str], event.get('log_system', None))
    if system is None:
        level = cast(Optional[NamedConstant], event.get('log_level', None))
        if level is None:
            levelName = '-'
        else:
            levelName = level.name
        system = '{namespace}#{level}'.format(namespace=cast(str, event.get('log_namespace', '-')), level=levelName)
    else:
        try:
            system = str(system)
        except Exception:
            system = 'UNFORMATTABLE'
    return system