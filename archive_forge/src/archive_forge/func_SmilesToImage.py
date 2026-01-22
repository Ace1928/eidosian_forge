import os
import sys
import tempfile
from PIL import Image
from rdkit import RDConfig
def SmilesToImage(smiles, **kwargs):
    with tempfile.NamedTemporaryFile(suffix='.gif') as tmp:
        ok = SmilesToGif(smiles, tmp.name, **kwargs)
        if ok:
            img = Image.open(tmp.name)
        else:
            img = None
    return img