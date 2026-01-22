import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class SymbolFunction(CommandBit):
    """Find a function which is represented by a symbol (like _ or ^)"""
    commandmap = FormulaConfig.symbolfunctions

    def detect(self, pos):
        """Find the symbol"""
        return pos.current() in SymbolFunction.commandmap

    def parsebit(self, pos):
        """Parse the symbol"""
        self.setcommand(pos.current())
        pos.skip(self.command)
        self.output = TaggedOutput().settag(self.translated)
        self.parseparameter(pos)