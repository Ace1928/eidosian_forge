import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FormulaConstant(Constant):
    """A constant string in a formula"""

    def __init__(self, string):
        """Set the constant string"""
        Constant.__init__(self, string)
        self.original = string
        self.size = 1
        self.type = None

    def computesize(self):
        """Compute the size of the constant: always 1."""
        return self.size

    def clone(self):
        """Return a copy of itself."""
        return FormulaConstant(self.original)

    def __unicode__(self):
        """Return a printable representation."""
        return 'Formula constant: ' + self.string