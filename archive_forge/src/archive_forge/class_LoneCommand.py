import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class LoneCommand(Parser):
    """A parser for just one command line"""

    def parse(self, reader):
        """Read nothing"""
        return []