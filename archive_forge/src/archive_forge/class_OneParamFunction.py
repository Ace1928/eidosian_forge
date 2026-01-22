import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class OneParamFunction(CommandBit):
    """A function of one parameter"""
    commandmap = FormulaConfig.onefunctions
    simplified = False

    def parsebit(self, pos):
        """Parse a function with one parameter"""
        self.output = TaggedOutput().settag(self.translated)
        self.parseparameter(pos)
        self.simplifyifpossible()

    def simplifyifpossible(self):
        """Try to simplify to a single character."""
        if self.original in self.commandmap:
            self.output = FixedOutput()
            self.html = [self.commandmap[self.original]]
            self.simplified = True