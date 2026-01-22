from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class PositionAttachCursiveDefinition(Statement):

    def __init__(self, coverages_exit, coverages_enter, location=None):
        Statement.__init__(self, location)
        self.coverages_exit = coverages_exit
        self.coverages_enter = coverages_enter

    def __str__(self):
        res = 'AS_POSITION\nATTACH_CURSIVE'
        for coverage in self.coverages_exit:
            coverage = ''.join((str(c) for c in coverage))
            res += f'\nEXIT {coverage}'
        for coverage in self.coverages_enter:
            coverage = ''.join((str(c) for c in coverage))
            res += f'\nENTER {coverage}'
        res += '\nEND_ATTACH\nEND_POSITION'
        return res