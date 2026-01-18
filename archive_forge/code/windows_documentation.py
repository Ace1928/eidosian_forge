import ctypes
from ctypes import wintypes
import platform
from pyu2f import errors
from pyu2f.hid import base
Closes the file handle when object is GC-ed.