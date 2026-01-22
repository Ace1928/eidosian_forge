import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _copyXyMouseInfo(self, *args):
    if len(args) > 0 and isinstance(args[0], Event):
        args = ()
    if self.delayEnabledSV.get() == 'on' and len(args) == 0:
        self.root.after(1000, self._copyXyMouseInfo, 2)
        self.xyCopyButtonSV.set('Copy in 3')
    elif len(args) == 1 and args[0] == 2:
        self.root.after(1000, self._copyXyMouseInfo, 1)
        self.xyCopyButtonSV.set('Copy in 2')
    elif len(args) == 1 and args[0] == 1:
        self.root.after(1000, self._copyXyMouseInfo, 0)
        self.xyCopyButtonSV.set('Copy in 1')
    else:
        self._copyText(self.xyTextboxSV.get())
        self.xyCopyButtonSV.set('Copy XY')