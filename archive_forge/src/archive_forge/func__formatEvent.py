from __future__ import annotations
from datetime import datetime as DateTime
from typing import Any, Callable, Iterator, Mapping, Optional, Union, cast
from constantly import NamedConstant
from twisted.python._tzhelper import FixedOffsetTimeZone
from twisted.python.failure import Failure
from twisted.python.reflect import safe_repr
from ._flatten import aFormatter, flatFormat
from ._interfaces import LogEvent
def _formatEvent(event: LogEvent) -> str:
    """
    Formats an event as a string, using the format in C{event["log_format"]}.

    This implementation should never raise an exception; if the formatting
    cannot be done, the returned string will describe the event generically so
    that a useful message is emitted regardless.

    @param event: A logging event.

    @return: A formatted string.
    """
    try:
        if 'log_flattened' in event:
            return flatFormat(event)
        format = cast(Optional[Union[str, bytes]], event.get('log_format', None))
        if format is None:
            return ''
        if isinstance(format, str):
            pass
        elif isinstance(format, bytes):
            format = format.decode('utf-8')
        else:
            raise TypeError(f'Log format must be str, not {format!r}')
        return formatWithCall(format, event)
    except BaseException as e:
        return formatUnformattableEvent(event, e)