import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
def addempty(self):
    """Add an empty row."""
    row = self.factory.create(FormulaRow).setalignments(self.alignments)
    for index, originalcell in enumerate(self.rows[-1].contents):
        cell = row.createcell(index)
        cell.add(FormulaConstant('\u2005'))
        row.add(cell)
    self.addrow(row)