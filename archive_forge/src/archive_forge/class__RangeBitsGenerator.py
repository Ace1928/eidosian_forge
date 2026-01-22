import random
from rdkit import Chem
from rdkit.Chem.rdDistGeom import EmbedMolecule
class _RangeBitsGenerator(object):

    def __init__(self, nCenters):
        self.nCenters = nCenters

    def __iter__(self):
        for val in range(2 ** self.nCenters):
            yield val