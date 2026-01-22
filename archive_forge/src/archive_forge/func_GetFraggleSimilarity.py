import sys
from itertools import combinations
from rdkit import Chem, DataStructs
from rdkit.Chem import rdqueries
def GetFraggleSimilarity(queryMol, refMol, tverskyThresh=0.8):
    """ return the Fraggle similarity between two molecules

    >>> q = Chem.MolFromSmiles('COc1cc(CN2CCC(NC(=O)c3cncc(C)c3)CC2)c(OC)c2ccccc12')
    >>> m = Chem.MolFromSmiles('COc1cc(CN2CCC(NC(=O)c3ccccc3)CC2)c(OC)c2ccccc12')
    >>> sim,match = GetFraggleSimilarity(q,m)
    >>> sim
    0.980...
    >>> match
    '*C1CCN(Cc2cc(OC)c3ccccc3c2OC)CC1'

    >>> m = Chem.MolFromSmiles('COc1cc(CN2CCC(Nc3nc4ccccc4s3)CC2)c(OC)c2ccccc12')
    >>> sim,match = GetFraggleSimilarity(q,m)
    >>> sim
    0.794...
    >>> match
    '*C1CCN(Cc2cc(OC)c3ccccc3c2OC)CC1'

    >>> q = Chem.MolFromSmiles('COc1ccccc1')
    >>> sim,match = GetFraggleSimilarity(q,m)
    >>> sim
    0.347...
    >>> match
    '*c1ccccc1'

    """
    if hasattr(queryMol, '_fraggleDecomp'):
        frags = queryMol._fraggleDecomp
    else:
        frags = generate_fraggle_fragmentation(queryMol)
        queryMol._fraggleDecomp = frags
    qSmi = Chem.MolToSmiles(queryMol, True)
    result = 0.0
    bestMatch = None
    for frag in frags:
        _, fragsim = compute_fraggle_similarity_for_subs(refMol, queryMol, qSmi, frag, tverskyThresh)
        if fragsim > result:
            result = fragsim
            bestMatch = frag
    return (result, bestMatch)