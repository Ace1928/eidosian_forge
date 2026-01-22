import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class AlphaCommand(EmptyCommand):
    """A command without paramters whose result is alphabetical"""
    commandmap = FormulaConfig.alphacommands

    def parsebit(self, pos):
        """Parse the command and set type to alpha"""
        EmptyCommand.parsebit(self, pos)
        self.type = 'alpha'