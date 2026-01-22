from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class FeatureDefinition(Statement):

    def __init__(self, name, tag, lookups, location=None):
        Statement.__init__(self, location)
        self.name = name
        self.tag = tag
        self.lookups = lookups

    def __str__(self):
        res = f'DEF_FEATURE NAME "{self.name}" TAG "{self.tag}"\n'
        res += ' ' + ' '.join((f'LOOKUP "{l}"' for l in self.lookups)) + '\n'
        res += 'END_FEATURE\n'
        return res