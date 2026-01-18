import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def traversewhole(self, formula):
    """Traverse over the contents to alter variables and space units."""
    last = None
    for bit, contents in self.traverse(formula):
        if bit.type == 'alpha':
            self.italicize(bit, contents)
        elif bit.type == 'font' and last and (last.type == 'number'):
            bit.contents.insert(0, FormulaConstant('\u205f'))
        last = bit