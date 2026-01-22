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
def _screenshot_osx(imageFilename=None, region=None):
    """
    TODO
    """
    if PILLOW_VERSION < (6, 2, 1):
        if imageFilename is None:
            tmpFilename = 'screenshot%s.png' % datetime.datetime.now().strftime('%Y-%m%d_%H-%M-%S-%f')
        else:
            tmpFilename = imageFilename
        subprocess.call(['screencapture', '-x', tmpFilename])
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
    elif region is not None:
        im = ImageGrab.grab(bbox=(region[0], region[1], region[2] + region[0], region[3] + region[1]))
    else:
        im = ImageGrab.grab()
    return im