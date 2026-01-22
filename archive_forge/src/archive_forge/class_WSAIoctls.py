from __future__ import annotations
import enum
import re
from typing import TYPE_CHECKING, NewType, NoReturn, Protocol, cast
import cffi
class WSAIoctls(enum.IntEnum):
    SIO_BASE_HANDLE = 1207959586
    SIO_BSP_HANDLE_SELECT = 1207959580
    SIO_BSP_HANDLE_POLL = 1207959581