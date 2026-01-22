import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class Bracket(FormulaBit):
    """A {} bracket inside a formula"""
    start = FormulaConfig.starts['bracket']
    ending = FormulaConfig.endings['bracket']

    def __init__(self):
        """Create a (possibly literal) new bracket"""
        FormulaBit.__init__(self)
        self.inner = None

    def detect(self, pos):
        """Detect the start of a bracket"""
        return pos.checkfor(self.start)

    def parsebit(self, pos):
        """Parse the bracket"""
        self.parsecomplete(pos, self.innerformula)
        return self

    def parsetext(self, pos):
        """Parse a text bracket"""
        self.parsecomplete(pos, self.innertext)
        return self

    def parseliteral(self, pos):
        """Parse a literal bracket"""
        self.parsecomplete(pos, self.innerliteral)
        return self

    def parsecomplete(self, pos, innerparser):
        """Parse the start and end marks"""
        if not pos.checkfor(self.start):
            Trace.error('Bracket should start with ' + self.start + ' at ' + pos.identifier())
            return None
        self.skiporiginal(self.start, pos)
        pos.pushending(self.ending)
        innerparser(pos)
        self.original += pos.popending(self.ending)
        self.computesize()

    def innerformula(self, pos):
        """Parse a whole formula inside the bracket"""
        while not pos.finished():
            self.add(self.factory.parseany(pos))

    def innertext(self, pos):
        """Parse some text inside the bracket, following textual rules."""
        specialchars = list(FormulaConfig.symbolfunctions.keys())
        specialchars.append(FormulaConfig.starts['command'])
        specialchars.append(FormulaConfig.starts['bracket'])
        specialchars.append(Comment.start)
        while not pos.finished():
            if pos.current() in specialchars:
                self.add(self.factory.parseany(pos))
                if pos.checkskip(' '):
                    self.original += ' '
            else:
                self.add(FormulaConstant(pos.skipcurrent()))

    def innerliteral(self, pos):
        """Parse a literal inside the bracket, which does not generate HTML."""
        self.literal = ''
        while not pos.finished() and (not pos.current() == self.ending):
            if pos.current() == self.start:
                self.parseliteral(pos)
            else:
                self.literal += pos.skipcurrent()
        self.original += self.literal