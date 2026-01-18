import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def writeparam(self, pos):
    """Write a single param of the form $0, $x..."""
    name = '$' + pos.skipcurrent()
    if not name in self.params:
        Trace.error('Unknown parameter ' + name)
        return None
    if not self.params[name]:
        return None
    if pos.checkskip('.'):
        self.params[name].value.type = pos.globalpha()
    return self.params[name].value