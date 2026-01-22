import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _linuxScreenshot(filename=None):
    if not scrotExists:
        raise NotImplementedError('"scrot" must be installed to use screenshot functions in Linux. Run: sudo apt-get install scrot')
    if filename is not None:
        tmpFilename = filename
    else:
        tmpFilename = '.screenshot%s.png' % datetime.datetime.now().strftime('%Y-%m%d_%H-%M-%S-%f')
    if scrotExists:
        subprocess.call(['scrot', '-z', tmpFilename])
        im = Image.open(tmpFilename)
        im.load()
        if filename is None:
            os.unlink(tmpFilename)
        return im
    else:
        raise Exception('The scrot program must be installed to take a screenshot with PyScreeze on Linux. Run: sudo apt-get install scrot')