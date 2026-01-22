import ctypes
import sys
from typing import Any
import time
from ctypes import Structure, byref, wintypes
from typing import IO, NamedTuple, Type, cast
from pip._vendor.rich.color import ColorSystem
from pip._vendor.rich.style import Style
def FillConsoleOutputCharacter(std_handle: wintypes.HANDLE, char: str, length: int, start: WindowsCoordinates) -> int:
    """Writes a character to the console screen buffer a specified number of times, beginning at the specified coordinates.

    Args:
        std_handle (wintypes.HANDLE): A handle to the console input buffer or the console screen buffer.
        char (str): The character to write. Must be a string of length 1.
        length (int): The number of times to write the character.
        start (WindowsCoordinates): The coordinates to start writing at.

    Returns:
        int: The number of characters written.
    """
    character = ctypes.c_char(char.encode())
    num_characters = wintypes.DWORD(length)
    num_written = wintypes.DWORD(0)
    _FillConsoleOutputCharacterW(std_handle, character, num_characters, start, byref(num_written))
    return num_written.value