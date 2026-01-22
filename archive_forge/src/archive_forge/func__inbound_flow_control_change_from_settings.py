from enum import Enum, IntEnum
from hpack import HeaderTuple
from hyperframe.frame import (
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .utilities import (
from .windows import WindowManager
def _inbound_flow_control_change_from_settings(self, delta):
    """
        We changed SETTINGS_INITIAL_WINDOW_SIZE, which means we need to
        update the target window size for flow control. For our flow control
        strategy, this means we need to do two things: we need to adjust the
        current window size, but we also need to set the target maximum window
        size to the new value.
        """
    new_max_size = self._inbound_window_manager.max_window_size + delta
    self._inbound_window_manager.window_opened(delta)
    self._inbound_window_manager.max_window_size = new_max_size