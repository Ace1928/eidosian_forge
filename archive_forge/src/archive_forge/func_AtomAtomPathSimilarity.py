import time
import unittest
import numpy
from scipy.optimize import linear_sum_assignment
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, rdmolops
from rdkit.Chem.Fingerprints import FingerprintMols
def AtomAtomPathSimilarity(m1, m2, m1pathintegers=None, m2pathintegers=None):
    """compute the Atom Atom Path Similarity for a pair of RDKit molecules.  See Gobbi et al, J. ChemInf (2015) 7:11
      the most expensive part of the calculation is computing the path integers - we can precompute these and pass them in as an argument"""
    if m1pathintegers is None:
        m1pathintegers = getpathintegers(m1)
    if m2pathintegers is None:
        m2pathintegers = getpathintegers(m2)
    simmatrix = getsimmatrix(m1, m1pathintegers, m2, m2pathintegers)
    mappings = gethungarianmappings(simmatrix)
    simab = getsimab(mappings, simmatrix)
    return simab