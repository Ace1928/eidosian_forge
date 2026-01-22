import base64
import hashlib
import logging
import os
import re
import tempfile
import uuid
from collections import namedtuple
from rdkit import Chem, RDConfig
from rdkit.Chem.MolKey import InchiInfo
def _get_identification_string(err, ctab, inchi, stereo_category=None, extra_stereo=None):
    if err & NULL_MOL:
        return _get_null_mol_identification_string(extra_stereo)
    elif err & BAD_SET:
        return _get_bad_mol_identification_string(ctab, stereo_category, extra_stereo)
    pieces = []
    if inchi:
        pieces.append(inchi)
    if not stereo_category:
        raise MolIdentifierException('Stereo category may not be left undefined')
    pieces.append(f'ST={stereo_category}')
    if extra_stereo:
        pieces.append(f'XTR={extra_stereo}')
    return '/'.join(pieces)