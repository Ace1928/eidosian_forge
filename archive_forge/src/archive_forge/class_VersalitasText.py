import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class VersalitasText(TaggedText):
    """Text in versalitas"""

    def process(self):
        self.output.tag = 'span class="versalitas"'