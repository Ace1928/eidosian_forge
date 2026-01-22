import base64
from enum import Enum, IntEnum
from hyperframe.exceptions import InvalidPaddingError
from hyperframe.frame import (
from hpack.hpack import Encoder, Decoder
from hpack.exceptions import HPACKError, OversizedHeaderListError
from .config import H2Configuration
from .errors import ErrorCodes, _error_code_from_int
from .events import (
from .exceptions import (
from .frame_buffer import FrameBuffer
from .settings import Settings, SettingCodes
from .stream import H2Stream, StreamClosedBy
from .utilities import SizeLimitDict, guard_increment_window
from .windows import WindowManager
def _local_settings_acked(self):
    """
        Handle the local settings being ACKed, update internal state.
        """
    changes = self.local_settings.acknowledge()
    if SettingCodes.INITIAL_WINDOW_SIZE in changes:
        setting = changes[SettingCodes.INITIAL_WINDOW_SIZE]
        self._inbound_flow_control_change_from_settings(setting.original_value, setting.new_value)
    if SettingCodes.MAX_HEADER_LIST_SIZE in changes:
        setting = changes[SettingCodes.MAX_HEADER_LIST_SIZE]
        self.decoder.max_header_list_size = setting.new_value
    if SettingCodes.MAX_FRAME_SIZE in changes:
        setting = changes[SettingCodes.MAX_FRAME_SIZE]
        self.max_inbound_frame_size = setting.new_value
    if SettingCodes.HEADER_TABLE_SIZE in changes:
        setting = changes[SettingCodes.HEADER_TABLE_SIZE]
        self.decoder.max_allowed_table_size = setting.new_value
    return changes