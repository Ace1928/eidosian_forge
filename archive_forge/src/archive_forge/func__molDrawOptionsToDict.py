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
def _molDrawOptionsToDict(molDrawOptions=rdMolDraw2D.MolDrawOptions()):
    return {key: _getValueFromKey(molDrawOptions, key) for key in dir(molDrawOptions) if _isAcceptedKeyValue(key, getattr(molDrawOptions, key))}