import math
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
def Dot(v1, v2):
    """ Returns the Dot product between two vectors:

    **Arguments**:

      - two vectors (sequences of bit ids)

    **Returns**: an integer

    **Notes**

      - the vectors must be sorted

      - duplicate bit IDs are counted more than once

    >>> Dot( (1,2,3,4,10), (2,4,6) )
    2

    Here's how duplicates are handled:

    >>> Dot( (1,2,2,3,4), (2,2,4,5,6) )
    5
    >>> Dot( (1,2,2,3,4), (2,4,5,6) )
    2
    >>> Dot( (1,2,2,3,4), (5,6) )
    0
    >>> Dot( (), (5,6) )
    0

    """
    res = 0
    nV1 = len(v1)
    nV2 = len(v2)
    i = 0
    j = 0
    while i < nV1:
        v1Val = v1[i]
        v1Count = 1
        i += 1
        while i < nV1 and v1[i] == v1Val:
            v1Count += 1
            i += 1
        while j < nV2 and v2[j] < v1Val:
            j += 1
        if j < nV2 and v2[j] == v1Val:
            v2Count = 1
            j += 1
            while j < nV2 and v2[j] == v1Val:
                v2Count += 1
                j += 1
            commonCount = min(v1Count, v2Count)
            res += commonCount * commonCount
        elif j >= nV2:
            break
    return res