from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union
from ._events import (
from ._headers import get_comma_header, has_expect_100_continue, set_comma_header
from ._readers import READERS, ReadersType
from ._receivebuffer import ReceiveBuffer
from ._state import (
from ._util import (  # Import the internal things we need
from ._writers import WRITERS, WritersType
def _extract_next_receive_event(self) -> Union[Event, Type[NEED_DATA], Type[PAUSED]]:
    state = self.their_state
    if state is DONE and self._receive_buffer:
        return PAUSED
    if state is MIGHT_SWITCH_PROTOCOL or state is SWITCHED_PROTOCOL:
        return PAUSED
    assert self._reader is not None
    event = self._reader(self._receive_buffer)
    if event is None:
        if not self._receive_buffer and self._receive_buffer_closed:
            if hasattr(self._reader, 'read_eof'):
                event = self._reader.read_eof()
            else:
                event = ConnectionClosed()
    if event is None:
        event = NEED_DATA
    return event