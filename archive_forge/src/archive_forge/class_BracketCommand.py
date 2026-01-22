import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class BracketCommand(OneParamFunction):
    """A command which defines a bracket."""
    commandmap = FormulaConfig.bracketcommands

    def parsebit(self, pos):
        """Parse the bracket."""
        OneParamFunction.parsebit(self, pos)

    def create(self, direction, character):
        """Create the bracket for the given character."""
        self.original = character
        self.command = '\\' + direction
        self.contents = [FormulaConstant(character)]
        return self