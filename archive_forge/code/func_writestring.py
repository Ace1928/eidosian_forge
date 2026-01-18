import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def writestring(self, string):
    """Write a string"""
    if not self.file:
        self.file = codecs.open(self.filename, 'w', 'utf-8')
    if self.file == sys.stdout and sys.version_info < (3, 0):
        string = string.encode('utf-8')
    self.file.write(string)