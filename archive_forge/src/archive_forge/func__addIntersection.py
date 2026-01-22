from fontTools.pens.basePen import BasePen
from fontTools.misc.bezierTools import solveQuadratic, solveCubic
def _addIntersection(self, goingUp):
    if self.evenOdd or goingUp:
        self.intersectionCount += 1
    else:
        self.intersectionCount -= 1