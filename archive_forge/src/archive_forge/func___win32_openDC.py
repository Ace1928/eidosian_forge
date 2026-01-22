import collections
import datetime
import functools
import os
import subprocess
import sys
import time
import errno
from contextlib import contextmanager
from PIL import Image
from PIL import ImageOps
from PIL import ImageDraw
from PIL import __version__ as PIL__version__
from PIL import ImageGrab
@contextmanager
def __win32_openDC(hWnd=0):
    """
        A context manager for handling calling GetDC() and ReleaseDC().

        This is used for win32 API calls, used by the pixel() function
        on Windows.

        Args:
            hWnd (int): The handle for the window to get a device context
        of, defaults to 0
        """
    hDC = windll.user32.GetDC(hWnd)
    if hDC == 0:
        raise WindowsError('windll.user32.GetDC failed : return NULL')
    try:
        yield hDC
    finally:
        windll.user32.ReleaseDC.argtypes = [ctypes.c_ssize_t, ctypes.c_ssize_t]
        if windll.user32.ReleaseDC(hWnd, hDC) == 0:
            raise WindowsError('windll.user32.ReleaseDC failed : return 0')