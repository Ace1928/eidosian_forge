import sys
import os
import os.path
import ctypes
from ctypes import c_char_p, c_int, c_size_t, c_void_p, pointer, CFUNCTYPE, POINTER
import ctypes.util
import platform
import textwrap
Factory for generating callback function prototypes.

    This is factored into a factory so I can easily change the definition
    for experimentation or debugging.
    