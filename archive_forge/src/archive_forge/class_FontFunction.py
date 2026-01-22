import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FontFunction(OneParamFunction):
    """A function of one parameter that changes the font"""
    commandmap = FormulaConfig.fontfunctions

    def process(self):
        """Simplify if possible using a single character."""
        self.type = 'font'
        self.simplifyifpossible()