import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.Polypeptide import PPBuilder
def _make_fragment_list(pp, length):
    """Dice up a peptide in fragments of length "length" (PRIVATE).

    :param pp: a list of residues (part of one peptide)
    :type pp: [L{Residue}, L{Residue}, ...]

    :param length: fragment length
    :type length: int
    """
    frag_list = []
    for i in range(len(pp) - length + 1):
        f = Fragment(length, -1)
        for j in range(length):
            residue = pp[i + j]
            resname = residue.get_resname()
            if residue.has_id('CA'):
                ca = residue['CA']
            else:
                raise PDBException('CHAINBREAK')
            if ca.is_disordered():
                raise PDBException('CHAINBREAK')
            ca_coord = ca.get_coord()
            f.add_residue(resname, ca_coord)
        frag_list.append(f)
    return frag_list