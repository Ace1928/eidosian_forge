import enum
import hashlib
import json
import logging
import re
from typing import Iterable, Optional
from rdkit import Chem
from rdkit.Chem import rdMolHash
class HashLayer(enum.Enum):
    """
    :cvar CANONICAL_SMILES: RDKit canonical SMILES (excluding enhanced stereo)
    :cvar ESCAPE: arbitrary other information to be incorporated
    :cvar FORMULA: a simple molecular formula for the molecule
    :cvar NO_STEREO_SMILES: RDKit canonical SMILES with all stereo removed
    :cvar SGROUP_DATA: canonicalization of all SGroups data present
    :cvar TAUTOMER_HASH: SMILES-like representation for a generic tautomer form
    :cvar NO_STEREO_TAUTOMER_HASH: the above tautomer hash lacking all stereo
    """
    CANONICAL_SMILES = enum.auto()
    ESCAPE = enum.auto()
    FORMULA = enum.auto()
    NO_STEREO_SMILES = enum.auto()
    NO_STEREO_TAUTOMER_HASH = enum.auto()
    SGROUP_DATA = enum.auto()
    TAUTOMER_HASH = enum.auto()