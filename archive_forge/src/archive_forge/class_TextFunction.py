import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class TextFunction(CommandBit):
    """A function where parameters are read as text."""
    commandmap = FormulaConfig.textfunctions

    def parsebit(self, pos):
        """Parse a text parameter"""
        self.output = TaggedOutput().settag(self.translated)
        self.parsetext(pos)

    def process(self):
        """Set the type to font"""
        self.type = 'font'