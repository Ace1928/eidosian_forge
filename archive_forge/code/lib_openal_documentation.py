import ctypes
from ctypes import *
import pyglet.lib
Wrapper for openal

Generated with:
../tools/wraptypes/wrap.py /usr/include/AL/al.h -lopenal -olib_openal.py

.. Hacked to remove non-existent library functions.

TODO add alGetError check.

.. alListener3i and alListeneriv are present in my OS X 10.4 but not another
10.4 user's installation.  They've also been removed for compatibility.
