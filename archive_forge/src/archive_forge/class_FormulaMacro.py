import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FormulaMacro(Formula):
    """A math macro defined in an inset."""

    def __init__(self):
        self.parser = MacroParser()
        self.output = EmptyOutput()

    def __unicode__(self):
        """Return a printable representation."""
        return 'Math macro'