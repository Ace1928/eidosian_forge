from typing import Iterable, Sequence, Tuple, cast
from pip._vendor.rich._win32_console import LegacyWindowsTerm, WindowsCoordinates
from pip._vendor.rich.segment import ControlCode, ControlType, Segment
Makes appropriate Windows Console API calls based on the segments in the buffer.

    Args:
        buffer (Iterable[Segment]): Iterable of Segments to convert to Win32 API calls.
        term (LegacyWindowsTerm): Used to call the Windows Console API.
    