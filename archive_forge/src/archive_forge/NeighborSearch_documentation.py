import numpy as np
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.Selection import unfold_entities, entity_levels, uniqueify
All neighbor search.

        Search all entities that have atoms pairs within
        radius.

        Arguments:
         - radius - float
         - level - char (A, R, C, M, S)

        