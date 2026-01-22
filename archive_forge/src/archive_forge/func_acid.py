from warnings import warn
import copy
import logging
from rdkit import Chem
from .utils import memoized_property
@memoized_property
def acid(self):
    log.debug(f'Loading AcidBasePair acid: {self.name}')
    return Chem.MolFromSmarts(self.acid_str)