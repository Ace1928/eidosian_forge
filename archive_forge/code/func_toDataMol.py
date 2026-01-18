import base64
import json
import logging
import re
import uuid
from xml.dom import minidom
from IPython.display import HTML, display
from rdkit import Chem
from rdkit.Chem import Draw
from . import rdMolDraw2D
def toDataMol(mol):
    return 'pkl_' + base64.b64encode(mol.ToBinary(Chem.PropertyPickleOptions.AllProps ^ Chem.PropertyPickleOptions.ComputedProps)).decode('utf-8')