import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class LyXLine(Container):
    """A Lyx line"""

    def __init__(self):
        self.parser = LoneCommand()
        self.output = FixedOutput()

    def process(self):
        self.html = ['<hr class="line" />']