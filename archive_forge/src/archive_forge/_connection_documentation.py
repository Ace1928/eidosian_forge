from typing import Any, Callable, cast, Dict, List, Optional, Tuple, Type, Union
from ._events import (
from ._headers import get_comma_header, has_expect_100_continue, set_comma_header
from ._readers import READERS, ReadersType
from ._receivebuffer import ReceiveBuffer
from ._state import (
from ._util import (  # Import the internal things we need
from ._writers import WRITERS, WritersType
Notify the state machine that we failed to send the data it gave
        us.

        This causes :attr:`Connection.our_state` to immediately become
        :data:`ERROR` -- see :ref:`error-handling` for discussion.

        