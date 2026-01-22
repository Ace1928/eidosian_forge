import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _copyAllMouseInfo(self, *args):
    textFieldContents = '%s %s %s' % (self.xyTextboxSV.get(), self.rgbSV.get(), self.rgbHexSV.get())
    if len(args) > 0 and isinstance(args[0], Event):
        args = ()
    if self.delayEnabledSV.get() == 'on' and len(args) == 0:
        self.root.after(1000, self._copyAllMouseInfo, 2)
        self.allCopyButtonSV.set('Copy in 3')
    elif len(args) == 1 and args[0] == 2:
        self.root.after(1000, self._copyAllMouseInfo, 1)
        self.allCopyButtonSV.set('Copy in 2')
    elif len(args) == 1 and args[0] == 1:
        self.root.after(1000, self._copyAllMouseInfo, 0)
        self.allCopyButtonSV.set('Copy in 1')
    else:
        textFieldContents = '%s %s %s' % (self.xyTextboxSV.get(), self.rgbSV.get(), self.rgbHexSV.get())
        self._copyText(textFieldContents)
        self.allCopyButtonSV.set('Copy All')