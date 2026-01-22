import copy
import functools
import math
import numpy
from rdkit import Chem
def _drawBond(self, bond, atom, nbr, pos, nbrPos, conf, width=None, color=None, color2=None, labelSize1=None, labelSize2=None):
    width = width or self.drawingOptions.bondLineWidth
    color = color or self.drawingOptions.defaultColor
    color2 = color2 or self.drawingOptions.defaultColor
    p1_raw = copy.deepcopy(pos)
    p2_raw = copy.deepcopy(nbrPos)
    newpos = self._getBondAttachmentCoordinates(p1_raw, p2_raw, labelSize1)
    newnbrPos = self._getBondAttachmentCoordinates(p2_raw, p1_raw, labelSize2)
    addDefaultLine = functools.partial(self.canvas.addCanvasLine, linewidth=width, color=color, color2=color2)
    bType = bond.GetBondType()
    if bType == Chem.BondType.SINGLE:
        bDir = bond.GetBondDir()
        if bDir in (Chem.BondDir.BEGINWEDGE, Chem.BondDir.BEGINDASH):
            if bond.GetBeginAtom().GetChiralTag() in (Chem.ChiralType.CHI_TETRAHEDRAL_CW, Chem.ChiralType.CHI_TETRAHEDRAL_CCW):
                p1, p2 = (newpos, newnbrPos)
                wcolor = color
            else:
                p2, p1 = (newpos, newnbrPos)
                wcolor = color2
            if bDir == Chem.BondDir.BEGINWEDGE:
                self._drawWedgedBond(bond, p1, p2, color=wcolor, width=width)
            elif bDir == Chem.BondDir.BEGINDASH:
                self._drawWedgedBond(bond, p1, p2, color=wcolor, width=width, dash=self.drawingOptions.dash)
        else:
            addDefaultLine(newpos, newnbrPos)
    elif bType == Chem.BondType.DOUBLE:
        crossBond = self.drawingOptions.showUnknownDoubleBonds and bond.GetStereo() == Chem.BondStereo.STEREOANY
        if not crossBond and (bond.IsInRing() or (atom.GetDegree() != 1 and bond.GetOtherAtom(atom).GetDegree() != 1)):
            addDefaultLine(newpos, newnbrPos)
            fp1, fp2 = self._offsetDblBond(newpos, newnbrPos, bond, atom, nbr, conf)
            addDefaultLine(fp1, fp2)
        else:
            fp1, fp2 = self._offsetDblBond(newpos, newnbrPos, bond, atom, nbr, conf, direction=0.5, lenFrac=1.0)
            fp3, fp4 = self._offsetDblBond(newpos, newnbrPos, bond, atom, nbr, conf, direction=-0.5, lenFrac=1.0)
            if crossBond:
                fp2, fp4 = (fp4, fp2)
            addDefaultLine(fp1, fp2)
            addDefaultLine(fp3, fp4)
    elif bType == Chem.BondType.AROMATIC:
        addDefaultLine(newpos, newnbrPos)
        fp1, fp2 = self._offsetDblBond(newpos, newnbrPos, bond, atom, nbr, conf)
        addDefaultLine(fp1, fp2, dash=self.drawingOptions.dash)
    elif bType == Chem.BondType.TRIPLE:
        addDefaultLine(newpos, newnbrPos)
        fp1, fp2 = self._offsetDblBond(newpos, newnbrPos, bond, atom, nbr, conf)
        addDefaultLine(fp1, fp2)
        fp1, fp2 = self._offsetDblBond(newpos, newnbrPos, bond, atom, nbr, conf, direction=-1)
        addDefaultLine(fp1, fp2)
    else:
        addDefaultLine(newpos, newnbrPos, dash=(1, 2))