import math
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
def BitsInCommon(v1, v2):
    """ Returns the number of bits in common between two vectors

    **Arguments**:

      - two vectors (sequences of bit ids)

    **Returns**: an integer

    **Notes**

      - the vectors must be sorted

      - duplicate bit IDs are counted more than once

    >>> BitsInCommon( (1,2,3,4,10), (2,4,6) )
    2

    Here's how duplicates are handled:

    >>> BitsInCommon( (1,2,2,3,4), (2,2,4,5,6) )
    3

    """
    res = 0
    v2Pos = 0
    nV2 = len(v2)
    for val in v1:
        while v2Pos < nV2 and v2[v2Pos] < val:
            v2Pos += 1
        if v2Pos >= nV2:
            break
        if v2[v2Pos] == val:
            res += 1
            v2Pos += 1
    return res