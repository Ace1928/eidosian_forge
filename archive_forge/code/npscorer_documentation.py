import gzip
import math
import os.path
import pickle
import sys
from collections import namedtuple
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
Calculates the Natural Product Likeness of a molecule.

  Returns the score as float in the range -5..5.