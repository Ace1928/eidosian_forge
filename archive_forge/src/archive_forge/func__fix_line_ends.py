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
def _fix_line_ends(my_string):
    pat = '\r\n{0,1}'
    sbt = '\n'
    return _fix_all(pat, sbt, my_string)