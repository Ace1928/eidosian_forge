import pyperclip, sys, os, platform, webbrowser
from ctypes import (
import datetime, subprocess
def _copyText(self, textToCopy):
    try:
        pyperclip.copy(textToCopy)
        self.statusbarSV.set('Copied ' + textToCopy)
    except pyperclip.PyperclipException as e:
        if platform.system() == 'Linux':
            self.statusbarSV.set('Copy failed. Run "sudo apt-get install xsel".')
        else:
            self.statusbarSV.set('Clipboard error: ' + str(e))