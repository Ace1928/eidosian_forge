import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class BeginCommand(CommandBit):
    """A \\begin{}...\\end command and what it entails (array, cases, aligned)"""
    commandmap = {FormulaConfig.array['begin']: ''}
    types = [FormulaEquation, FormulaArray, FormulaCases, FormulaMatrix]

    def parsebit(self, pos):
        """Parse the begin command"""
        command = self.parseliteral(pos)
        bit = self.findbit(command)
        ending = FormulaConfig.array['end'] + '{' + command + '}'
        pos.pushending(ending)
        bit.parsebit(pos)
        self.add(bit)
        self.original += pos.popending(ending)
        self.size = bit.size

    def findbit(self, piece):
        """Find the command bit corresponding to the \\begin{piece}"""
        for type in BeginCommand.types:
            if piece.replace('*', '') == type.piece:
                return self.factory.create(type)
        bit = self.factory.create(EquationEnvironment)
        bit.piece = piece
        return bit