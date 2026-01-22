import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _logAllMouseInfo(self, *args):
    if len(args) > 0 and isinstance(args[0], Event):
        args = ()
    if self.delayEnabledSV.get() == 'on' and len(args) == 0:
        self.root.after(1000, self._logAllMouseInfo, 2)
        self.allLogButtonSV.set('Log in 3')
    elif len(args) == 1 and args[0] == 2:
        self.root.after(1000, self._logAllMouseInfo, 1)
        self.allLogButtonSV.set('Log in 2')
    elif len(args) == 1 and args[0] == 1:
        self.root.after(1000, self._logAllMouseInfo, 0)
        self.allLogButtonSV.set('Log in 1')
    else:
        textFieldContents = '%s %s %s' % (self.xyTextboxSV.get(), self.rgbSV.get(), self.rgbHexSV.get())
        logContents = self.logTextarea.get('1.0', 'end-1c') + '%s\n' % textFieldContents
        self.logTextboxSV.set(logContents)
        self._setLogTextAreaContents(logContents)
        self.statusbarSV.set('Logged ' + textFieldContents)
        self.allLogButtonSV.set('Log All')