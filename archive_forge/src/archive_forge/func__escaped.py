import sys, os, unicodedata
import py
from py.builtin import text, bytes
def _escaped(self, text, esc):
    if esc and self.hasmarkup:
        text = ''.join(['\x1b[%sm' % cod for cod in esc]) + text + '\x1b[0m'
    return text