import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FlexURL(URL):
    """A flexible URL"""

    def process(self):
        """Read URL from elyxer.contents"""
        self.url = self.extracttext()