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
def _get_chiral_identification_string(n_def, n_udf):
    assert n_def >= 0
    assert n_udf >= 0
    if n_def == 0:
        if n_udf == 0:
            return 'S_ACHIR'
        elif n_udf == 1:
            return 'R_ONE'
        else:
            return 'S_UNKN'
    elif n_udf == 0:
        return 'S_ABS'
    else:
        return 'S_PART'
    return 'OTHER'