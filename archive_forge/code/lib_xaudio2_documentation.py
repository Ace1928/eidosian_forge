import ctypes
import platform
import os
from pyglet.libs.win32.constants import *
from pyglet.libs.win32.types import *
from pyglet.libs.win32 import com
from pyglet.util import debug_print
This will attempt to load a version of XAudio2. Versions supported: 2.9, 2.8.
       While Windows 8 ships with 2.8 and Windows 10 ships with version 2.9, it is possible to install 2.9 on 8/8.1.
    