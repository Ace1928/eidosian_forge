import pickle
import re
from rdkit.Chem import Descriptors as DescriptorsMod
from rdkit.ML.Descriptors import Descriptors
from rdkit.RDLogger import logger
def _findVersions(self):
    """ returns a tuple of the versions of the descriptor calculators

    """
    self.descriptorVersions = []
    for nm in self.simpleList:
        vers = 'N/A'
        if hasattr(DescriptorsMod, nm):
            fn = getattr(DescriptorsMod, nm)
            if hasattr(fn, 'version'):
                vers = fn.version
        self.descriptorVersions.append(vers)