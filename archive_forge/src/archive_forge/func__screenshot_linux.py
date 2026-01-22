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
def _screenshot_linux(imageFilename=None, region=None):
    """
    TODO
    """
    if imageFilename is None:
        tmpFilename = '.screenshot%s.png' % datetime.datetime.now().strftime('%Y-%m%d_%H-%M-%S-%f')
    else:
        tmpFilename = imageFilename
    if PILLOW_VERSION >= (9, 2, 0) and GNOMESCREENSHOT_EXISTS:
        im = ImageGrab.grab()
        if imageFilename is not None:
            im.save(imageFilename)
        if region is None:
            return im
        else:
            assert len(region) == 4, 'region argument must be a tuple of four ints'
            assert isinstance(region[0], int) and isinstance(region[1], int) and isinstance(region[2], int) and isinstance(region[3], int), 'region argument must be a tuple of four ints'
            im = im.crop((region[0], region[1], region[2] + region[0], region[3] + region[1]))
            return im
    elif RUNNING_X11 and SCROT_EXISTS:
        subprocess.call(['scrot', '-z', tmpFilename])
    elif GNOMESCREENSHOT_EXISTS:
        subprocess.call(['gnome-screenshot', '-f', tmpFilename])
    elif RUNNING_WAYLAND and SCROT_EXISTS and (not GNOMESCREENSHOT_EXISTS):
        raise PyScreezeException('Your computer uses the Wayland window system. Scrot works on the X11 window system but not Wayland. You must install gnome-screenshot by running `sudo apt install gnome-screenshot`')
    else:
        raise Exception('To take screenshots, you must install Pillow version 9.2.0 or greater and gnome-screenshot by running `sudo apt install gnome-screenshot`')
    im = Image.open(tmpFilename)
    if region is not None:
        assert len(region) == 4, 'region argument must be a tuple of four ints'
        assert isinstance(region[0], int) and isinstance(region[1], int) and isinstance(region[2], int) and isinstance(region[3], int), 'region argument must be a tuple of four ints'
        im = im.crop((region[0], region[1], region[2] + region[0], region[3] + region[1]))
        os.unlink(tmpFilename)
        im.save(tmpFilename)
    else:
        im.load()
    if imageFilename is None:
        os.unlink(tmpFilename)
    return im