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
def ErrorBitsToText(err):
    """ returns a list of error bit descriptions for the error code provided """
    return [k for k, v in ERROR_DICT.items() if err & v > 0]