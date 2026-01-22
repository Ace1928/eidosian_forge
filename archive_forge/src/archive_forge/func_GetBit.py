from rdkit.Chem.Pharm2D import Matcher, SigFactory
def GetBit(self, idx):
    """ returns a bool indicating whether or not the bit is set

    """
    if idx < 0 or idx >= self.sig.GetSize():
        raise IndexError('Index %d invalid' % idx)
    if self.bits is not None and idx in self.bits:
        return self.bits[idx]
    tmp = Matcher.GetAtomsMatchingBit(self.sig, idx, self.mol, dMat=self.dMat, justOne=1, matchingAtoms=self.pattMatches)
    if not tmp or len(tmp) == 0:
        res = 0
    else:
        res = 1
    if self.bits is not None:
        self.bits[idx] = res
    return res