import random
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
class _BondFlipper(object):

    def __init__(self, bond):
        self.bond = bond

    def flip(self, flag):
        if flag:
            self.bond.SetStereo(Chem.BondStereo.STEREOCIS)
        else:
            self.bond.SetStereo(Chem.BondStereo.STEREOTRANS)