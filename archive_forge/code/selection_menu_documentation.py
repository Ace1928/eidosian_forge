import builtins
import sys
from ...utils.imports import _is_package_available
from . import cursor, input
from .helpers import Direction, clear_line, forceWrite, linebreak, move_cursor, reset_cursor, writeColor
from .keymap import KEYMAP
Start the menu and return the selected choice