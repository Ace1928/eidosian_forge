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
def _make_racemate_inchi(inchi):
    """ Normalize the stereo information (t-layer) to one selected isomer. """
    new_stereo = '/m0/s3/'
    stereo_match = GET_STEREO_RE.match(inchi)
    if stereo_match:
        inchi = stereo_match.group(1) + new_stereo + stereo_match.group(2)
    return inchi