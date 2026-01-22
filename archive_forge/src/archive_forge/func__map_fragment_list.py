import numpy as np
from Bio.SVDSuperimposer import SVDSuperimposer
from Bio.PDB.PDBExceptions import PDBException
from Bio.PDB.Polypeptide import PPBuilder
def _map_fragment_list(flist, reflist):
    """Map flist fragments to closest entry in reflist (PRIVATE).

    Map all frgaments in flist to the closest (in RMSD) fragment in reflist.

    Returns a list of reflist indices.

    :param flist: list of protein fragments
    :type flist: [L{Fragment}, L{Fragment}, ...]

    :param reflist: list of reference (ie. library) fragments
    :type reflist: [L{Fragment}, L{Fragment}, ...]
    """
    mapped = []
    for f in flist:
        rank = []
        for i in range(len(reflist)):
            rf = reflist[i]
            rms = f - rf
            rank.append((rms, rf))
        rank.sort()
        fragment = rank[0][1]
        mapped.append(fragment)
    return mapped