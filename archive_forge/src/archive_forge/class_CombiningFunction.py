import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class CombiningFunction(OneParamFunction):
    commandmap = FormulaConfig.combiningfunctions

    def parsebit(self, pos):
        """Parse a combining function."""
        self.type = 'alpha'
        combining = self.translated
        parameter = self.parsesingleparameter(pos)
        if not parameter:
            Trace.error('Empty parameter for combining function ' + self.command)
        elif len(parameter.extracttext()) != 1:
            Trace.error('Applying combining function ' + self.command + ' to invalid string "' + parameter.extracttext() + '"')
        self.contents.append(Constant(combining))

    def parsesingleparameter(self, pos):
        """Parse a parameter, or a single letter."""
        self.factory.clearskipped(pos)
        if pos.finished():
            Trace.error('Error while parsing single parameter at ' + pos.identifier())
            return None
        if self.factory.detecttype(Bracket, pos) or self.factory.detecttype(FormulaCommand, pos):
            return self.parseparameter(pos)
        letter = FormulaConstant(pos.skipcurrent())
        self.add(letter)
        return letter