import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class SquareBracket(Bracket):
    """A [] bracket inside a formula"""
    start = FormulaConfig.starts['squarebracket']
    ending = FormulaConfig.endings['squarebracket']

    def clone(self):
        """Return a new square bracket with the same contents."""
        bracket = SquareBracket()
        bracket.contents = self.contents
        return bracket