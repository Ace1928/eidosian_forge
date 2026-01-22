import enum
import hashlib
import json
import logging
import re
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdMolHash
def _RemoveUnnecessaryHs(rdk_mol, preserve_stereogenic_hs=False):
    """
    removes hydrogens that are not necessary for the registration hash, and
    preserves hydrogen isotopes
    """
    remove_hs_params = Chem.RemoveHsParameters()
    remove_hs_params.updateExplicitCount = True
    remove_hs_params.removeDefiningBondStereo = not preserve_stereogenic_hs
    edited_mol = Chem.rdmolops.RemoveHs(rdk_mol, remove_hs_params, sanitize=False)
    edited_mol.UpdatePropertyCache(False)
    return edited_mol