import logging
import sys
from base64 import b64encode
import numpy as np
import rdkit
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Draw, SDWriter, rdchem
from rdkit.Chem.Scaffolds import MurckoScaffold
from io import BytesIO
from xml.dom import minidom
def ChangeMoleculeRendering(frame=None, renderer='image'):
    """Allows to change the rendering of the molecules between image and string
    representations.
    This serves two purposes: First it allows to avoid the generation of images if this is
    not desired and, secondly, it allows to enable image rendering for newly created dataframe
    that already contains molecules, without having to rerun the time-consuming
    AddMoleculeColumnToFrame. Note: this behaviour is, because some pandas methods, e.g. head()
    returns a new dataframe instance that uses the default pandas rendering (thus not drawing
    images for molecules) instead of the monkey-patched one.
    """
    if frame is None:
        return
    try:
        PandasPatcher.changeMoleculeRendering(frame, renderer)
    except NameError:
        log.warning('Failed to patch pandas - unable to change molecule rendering')