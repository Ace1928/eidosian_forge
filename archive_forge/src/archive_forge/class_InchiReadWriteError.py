import logging
from rdkit import RDLogger
from rdkit.Chem import rdinchi
class InchiReadWriteError(Exception):
    pass