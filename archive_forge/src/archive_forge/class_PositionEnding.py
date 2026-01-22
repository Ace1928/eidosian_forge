import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class PositionEnding(object):
    """An ending for a parsing position"""

    def __init__(self, ending, optional):
        self.ending = ending
        self.optional = optional

    def checkin(self, pos):
        """Check for the ending"""
        return pos.checkfor(self.ending)

    def __unicode__(self):
        """Printable representation"""
        string = 'Ending ' + self.ending
        if self.optional:
            string += ' (optional)'
        return string