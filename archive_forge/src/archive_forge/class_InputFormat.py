import os
import re
from collections import namedtuple
from contextlib import closing
from rdkit import Chem, RDConfig
from rdkit.Chem.rdmolfiles import SDMolSupplier, SmilesMolSupplier
class InputFormat:
    SMARTS = 'smarts'
    MOL = 'mol'
    SMILES = 'smiles'