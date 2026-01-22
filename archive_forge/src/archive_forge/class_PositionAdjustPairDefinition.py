from fontTools.voltLib.error import VoltLibError
from typing import NamedTuple
class PositionAdjustPairDefinition(Statement):

    def __init__(self, coverages_1, coverages_2, adjust_pair, location=None):
        Statement.__init__(self, location)
        self.coverages_1 = coverages_1
        self.coverages_2 = coverages_2
        self.adjust_pair = adjust_pair

    def __str__(self):
        res = 'AS_POSITION\nADJUST_PAIR\n'
        for coverage in self.coverages_1:
            coverage = ' '.join((str(c) for c in coverage))
            res += f' FIRST {coverage}'
        res += '\n'
        for coverage in self.coverages_2:
            coverage = ' '.join((str(c) for c in coverage))
            res += f' SECOND {coverage}'
        res += '\n'
        for (id_1, id_2), (pos_1, pos_2) in self.adjust_pair.items():
            res += f' {id_1} {id_2} BY{pos_1}{pos_2}\n'
        res += '\nEND_ADJUST\nEND_POSITION'
        return res