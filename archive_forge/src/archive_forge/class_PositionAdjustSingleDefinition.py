from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class PositionAdjustSingleDefinition(Statement):

    def __init__(self, adjust_single, location=None):
        Statement.__init__(self, location)
        self.adjust_single = adjust_single

    def __str__(self):
        res = 'AS_POSITION\nADJUST_SINGLE'
        for coverage, pos in self.adjust_single:
            coverage = ''.join((str(c) for c in coverage))
            res += f'{coverage} BY{pos}'
        res += '\nEND_ADJUST\nEND_POSITION'
        return res