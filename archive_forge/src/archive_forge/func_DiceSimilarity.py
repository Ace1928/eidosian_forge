import math
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
def DiceSimilarity(v1, v2, bounds=None):
    """ Implements the DICE similarity metric.
     This is the recommended metric in both the Topological torsions
     and Atom pairs papers.

    **Arguments**:

      - two vectors (sequences of bit ids)

    **Returns**: a float.

    **Notes**

      - the vectors must be sorted


    >>> DiceSimilarity( (1,2,3), (1,2,3) )
    1.0
    >>> DiceSimilarity( (1,2,3), (5,6) )
    0.0
    >>> DiceSimilarity( (1,2,3,4), (1,3,5,7) )
    0.5
    >>> DiceSimilarity( (1,2,3,4,5,6), (1,3) )
    0.5

    Note that duplicate bit IDs count multiple times:

    >>> DiceSimilarity( (1,1,3,4,5,6), (1,1) )
    0.5

    but only if they are duplicated in both vectors:

    >>> DiceSimilarity( (1,1,3,4,5,6), (1,) )==2./7
    True

    edge case

    >>> DiceSimilarity( (), () )
    0.0

    and bounds check

    >>> DiceSimilarity( (1,1,3,4), (1,1))
    0.666...
    >>> DiceSimilarity( (1,1,3,4), (1,1), bounds=0.3)
    0.666...
    >>> DiceSimilarity( (1,1,3,4), (1,1), bounds=0.33)
    0.666...
    >>> DiceSimilarity( (1,1,3,4,5,6), (1,1), bounds=0.34)
    0.0

    """
    denom = 1.0 * (len(v1) + len(v2))
    if not denom:
        res = 0.0
    else:
        if bounds and min(len(v1), len(v2)) / denom < bounds:
            numer = 0.0
        else:
            numer = 2.0 * BitsInCommon(v1, v2)
        res = numer / denom
    return res