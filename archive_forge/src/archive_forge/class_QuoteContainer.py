import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class QuoteContainer(Container):
    """A container for a pretty quote"""

    def __init__(self):
        self.parser = BoundedParser()
        self.output = FixedOutput()

    def process(self):
        """Process contents"""
        self.type = self.header[2]
        if not self.type in StyleConfig.quotes:
            Trace.error('Quote type ' + self.type + ' not found')
            self.html = ['"']
            return
        self.html = [StyleConfig.quotes[self.type]]