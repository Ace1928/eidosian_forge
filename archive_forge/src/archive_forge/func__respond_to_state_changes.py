from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union
from ._events import (
from ._headers import get_comma_header, has_expect_100_continue, set_comma_header
from ._readers import READERS, ReadersType
from ._receivebuffer import ReceiveBuffer
from ._state import (
from ._util import (  # Import the internal things we need
from ._writers import WRITERS, WritersType
def _respond_to_state_changes(self, old_states: Dict[Type[Sentinel], Type[Sentinel]], event: Optional[Event]=None) -> None:
    if self.our_state != old_states[self.our_role]:
        self._writer = self._get_io_object(self.our_role, event, WRITERS)
    if self.their_state != old_states[self.their_role]:
        self._reader = self._get_io_object(self.their_role, event, READERS)