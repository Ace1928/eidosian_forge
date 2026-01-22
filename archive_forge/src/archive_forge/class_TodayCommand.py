import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class TodayCommand(EmptyCommand):
    """Shows today's date."""
    commandmap = None

    def parsebit(self, pos):
        """Parse a command without parameters"""
        self.output = FixedOutput()
        self.html = [datetime.date.today().strftime('%b %d, %Y')]