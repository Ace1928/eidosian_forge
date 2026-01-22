import sys
import os.path
import sys
import urllib.request, urllib.parse, urllib.error
import sys
import codecs
import unicodedata
import gettext
import datetime
class FormulaArray(MultiRowFormula):
    """An array within a formula"""
    piece = 'array'

    def parsebit(self, pos):
        """Parse the array"""
        self.output = TaggedOutput().settag('span class="array"', False)
        self.parsealignments(pos)
        self.parserows(pos)

    def parsealignments(self, pos):
        """Parse the different alignments"""
        self.valign = 'c'
        literal = self.parsesquareliteral(pos)
        if literal:
            self.valign = literal
        literal = self.parseliteral(pos)
        self.alignments = []
        for l in literal:
            self.alignments.append(l)