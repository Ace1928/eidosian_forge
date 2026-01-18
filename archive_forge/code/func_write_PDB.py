import re
from itertools import zip_longest
import numpy as np
from Bio.PDB.PDBExceptions import PDBException
from io import StringIO
from Bio.File import as_handle
from Bio.PDB.PDBIO import PDBIO
from Bio.PDB.Structure import Structure
from Bio.PDB.internal_coords import IC_Residue
from Bio.PDB.PICIO import write_PIC, read_PIC, enumerate_atoms, pdb_date
from typing import Dict, Union, Any, Tuple
from Bio.PDB.Atom import Atom
from Bio.PDB.Residue import Residue, DisorderedResidue
from Bio.PDB.Model import Model
from Bio.PDB.Chain import Chain
def write_PDB(entity: Structure, file: str, pdbid: str=None, chainid: str=None) -> None:
    """Write PDB file with HEADER and TITLE if available."""
    enumerate_atoms(entity)
    with as_handle(file, 'w') as fp:
        try:
            if hasattr(entity, 'header'):
                if not pdbid:
                    pdbid = entity.header.get('idcode', None)
                hdr = entity.header.get('head', None)
                dd = pdb_date(entity.header.get('deposition_date', None))
                if hdr:
                    fp.write('HEADER    {:40}{:8}   {:4}\n'.format(hdr.upper(), dd or '', pdbid or ''))
                nam = entity.header.get('name', None)
                if nam:
                    fp.write('TITLE     ' + nam.upper() + '\n')
            io = PDBIO()
            io.set_structure(entity)
            io.save(fp, preserve_atom_numbering=True)
        except KeyError:
            raise Exception('write_PDB: argument is not a Biopython PDB Entity ' + str(entity))