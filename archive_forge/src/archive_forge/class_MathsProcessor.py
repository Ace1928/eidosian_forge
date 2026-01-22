import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class MathsProcessor(object):
    """A processor for a maths construction inside the FormulaProcessor."""

    def process(self, contents, index):
        """Process an element inside a formula."""
        Trace.error('Unimplemented process() in ' + str(self))

    def __unicode__(self):
        """Return a printable description."""
        return 'Maths processor ' + self.__class__.__name__