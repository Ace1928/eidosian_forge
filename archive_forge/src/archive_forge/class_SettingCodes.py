import collections
import enum
from hyperframe.frame import SettingsFrame
from h2.errors import ErrorCodes
from h2.exceptions import InvalidSettingsValueError
class SettingCodes(enum.IntEnum):
    """
    All known HTTP/2 setting codes.

    .. versionadded:: 2.6.0
    """
    HEADER_TABLE_SIZE = SettingsFrame.HEADER_TABLE_SIZE
    ENABLE_PUSH = SettingsFrame.ENABLE_PUSH
    MAX_CONCURRENT_STREAMS = SettingsFrame.MAX_CONCURRENT_STREAMS
    INITIAL_WINDOW_SIZE = SettingsFrame.INITIAL_WINDOW_SIZE
    MAX_FRAME_SIZE = SettingsFrame.MAX_FRAME_SIZE
    MAX_HEADER_LIST_SIZE = SettingsFrame.MAX_HEADER_LIST_SIZE
    ENABLE_CONNECT_PROTOCOL = SettingsFrame.ENABLE_CONNECT_PROTOCOL