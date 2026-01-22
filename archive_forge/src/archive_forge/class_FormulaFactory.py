import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FormulaFactory(object):
    """Construct bits of formula"""
    types = [FormulaSymbol, RawText, FormulaNumber, Bracket, Comment, WhiteSpace]
    skippedtypes = [Comment, WhiteSpace]
    defining = False

    def __init__(self):
        """Initialize the map of instances."""
        self.instances = dict()

    def detecttype(self, type, pos):
        """Detect a bit of a given type."""
        if pos.finished():
            return False
        return self.instance(type).detect(pos)

    def instance(self, type):
        """Get an instance of the given type."""
        if not type in self.instances or not self.instances[type]:
            self.instances[type] = self.create(type)
        return self.instances[type]

    def create(self, type):
        """Create a new formula bit of the given type."""
        return Cloner.create(type).setfactory(self)

    def clearskipped(self, pos):
        """Clear any skipped types."""
        while not pos.finished():
            if not self.skipany(pos):
                return
        return

    def skipany(self, pos):
        """Skip any skipped types."""
        for type in self.skippedtypes:
            if self.instance(type).detect(pos):
                return self.parsetype(type, pos)
        return None

    def parseany(self, pos):
        """Parse any formula bit at the current location."""
        for type in self.types + self.skippedtypes:
            if self.detecttype(type, pos):
                return self.parsetype(type, pos)
        Trace.error('Unrecognized formula at ' + pos.identifier())
        return FormulaConstant(pos.skipcurrent())

    def parsetype(self, type, pos):
        """Parse the given type and return it."""
        bit = self.instance(type)
        self.instances[type] = None
        returnedbit = bit.parsebit(pos)
        if returnedbit:
            return returnedbit.setfactory(self)
        return bit

    def parseformula(self, formula):
        """Parse a string of text that contains a whole formula."""
        pos = TextPosition(formula)
        whole = self.create(WholeFormula)
        if whole.detect(pos):
            whole.parsebit(pos)
            return whole
        if not pos.finished():
            Trace.error('Unknown formula at: ' + pos.identifier())
            whole.add(TaggedBit().constant(formula, 'span class="unknown"'))
        return whole