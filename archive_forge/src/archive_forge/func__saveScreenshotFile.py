import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _saveScreenshotFile(self, *args):
    if not _PILLOW_INSTALLED:
        self.statusbarSV.set('ERROR: NA_Pillow_unsupported')
        return
    try:
        screenshot(self.screenshotFilenameSV.get())
    except Exception as e:
        self.statusbarSV.set('ERROR: ' + str(e))
    else:
        self.statusbarSV.set('Screenshot file saved to ' + self.screenshotFilenameSV.get())