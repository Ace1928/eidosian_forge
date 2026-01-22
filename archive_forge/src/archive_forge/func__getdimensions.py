import sys, os, unicodedata
import py
from py.builtin import text, bytes
def _getdimensions():
    handle = GetStdHandle(STD_OUTPUT_HANDLE)
    info = GetConsoleInfo(handle)
    return (info.dwSize.Y, info.dwSize.X - 1)