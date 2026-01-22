from Bio.PDB.Entity import Entity
from Bio.PDB.internal_coords import IC_Chain
from typing import Optional
Create/update atom coordinates from internal coordinates.

        :param verbose bool: default False
            describe runtime problems
        :param: start, fin integers
            optional sequence positions for begin, end of subregion to process.
            N.B. this activates serial residue assembly, <start> residue CA will
            be at origin
        :raises Exception: if any chain does not have .internal_coord attribute
        