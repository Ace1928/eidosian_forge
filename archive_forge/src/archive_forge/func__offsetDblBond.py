import copy
import functools
import math
import numpy
from rdkit import Chem
def _offsetDblBond(self, p1, p2, bond, a1, a2, conf, direction=1, lenFrac=None):
    perp, offsetX, offsetY = self._getBondOffset(p1, p2)
    offsetX = offsetX * direction
    offsetY = offsetY * direction
    if bond.IsInRing():
        bondIdx = bond.GetIdx()
        a2Idx = a2.GetIdx()
        for otherBond in a1.GetBonds():
            if otherBond.GetIdx() != bondIdx and otherBond.IsInRing():
                sharedRing = False
                for ring in self.bondRings:
                    if bondIdx in ring and otherBond.GetIdx() in ring:
                        sharedRing = True
                        break
                if not sharedRing:
                    continue
                a3 = otherBond.GetOtherAtom(a1)
                if a3.GetIdx() != a2Idx:
                    p3 = self.transformPoint(conf.GetAtomPosition(a3.GetIdx()) * self.drawingOptions.coordScale)
                    dx2 = p3[0] - p1[0]
                    dy2 = p3[1] - p1[1]
                    dotP = dx2 * offsetX + dy2 * offsetY
                    if dotP < 0:
                        perp += math.pi
                        offsetX = math.cos(perp) * self.drawingOptions.dblBondOffset * self.currDotsPerAngstrom
                        offsetY = math.sin(perp) * self.drawingOptions.dblBondOffset * self.currDotsPerAngstrom
    fracP1, fracP2 = self._getOffsetBondPts(p1, p2, offsetX, offsetY, lenFrac=lenFrac)
    return (fracP1, fracP2)