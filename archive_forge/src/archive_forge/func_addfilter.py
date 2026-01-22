import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def addfilter(self, index, value):
    """Add a filter for the given parameter number and parameter value."""
    original = '#' + str(index + 1)
    value = ''.join(self.values[0].gethtml())
    self.output.addfilter(original, value)