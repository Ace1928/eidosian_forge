import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _saveLogFile(self, *args):
    try:
        with open(self.logFilenameSV.get(), 'w') as fo:
            fo.write(self.logTextboxSV.get())
    except Exception as e:
        self.statusbarSV.set('ERROR: ' + str(e))
    else:
        self.statusbarSV.set('Log file saved to ' + self.logFilenameSV.get())