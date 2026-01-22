import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _updateMouseInfoTextFields(self):
    x, y = position()
    self.xyTextboxSV.set('%s,%s' % (x - self.xOrigin, y - self.yOrigin))
    width, height = size()
    if not _PILLOW_INSTALLED:
        self.rgbSV.set('NA_Pillow_unsupported')
    elif sys.platform == 'darwin':
        self.rgbSV.set('NA_on_macOS')
    elif not (0 <= x < width and 0 <= y < height):
        self.rgbSV.set('NA_on_multimonitor_setups')
    else:
        r, g, b = getPixel(x, y)
        self.rgbSV.set('%s,%s,%s' % (r, g, b))
    if not _PILLOW_INSTALLED:
        self.rgbHexSV.set('NA_Pillow_unsupported')
    elif sys.platform == 'darwin':
        self.rgbHexSV.set('NA_on_macOS')
    elif not (0 <= x < width and 0 <= y < height):
        self.rgbHexSV.set('NA_on_multimonitor_setups')
    else:
        rHex = hex(r)[2:].upper().rjust(2, '0')
        gHex = hex(g)[2:].upper().rjust(2, '0')
        bHex = hex(b)[2:].upper().rjust(2, '0')
        hexColor = '#%s%s%s' % (rHex, gHex, bHex)
        self.rgbHexSV.set(hexColor)
    if not _PILLOW_INSTALLED or sys.platform == 'darwin' or (not (0 <= x < width and 0 <= y < height)):
        self.colorFrame.configure(background='black')
    else:
        self.colorFrame.configure(background=hexColor)
    if self.isRunning:
        self._updateMouseInfoJob = self.root.after(100, self._updateMouseInfoTextFields)
    else:
        return