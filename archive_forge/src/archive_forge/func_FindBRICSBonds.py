import copy
import re
import sys
from rdkit import Chem
from rdkit import RDRandom as random
from rdkit.Chem import rdChemReactions as Reactions
def FindBRICSBonds(mol, randomizeOrder=False, silent=True):
    """ returns the bonds in a molecule that BRICS would cleave

    >>> from rdkit import Chem
    >>> m = Chem.MolFromSmiles('CCCOCC')
    >>> res = list(FindBRICSBonds(m))
    >>> res
    [((3, 2), ('3', '4')), ((3, 4), ('3', '4'))]

    a more complicated case:

    >>> m = Chem.MolFromSmiles('CCCOCCC(=O)c1ccccc1')
    >>> res = list(FindBRICSBonds(m))
    >>> res
    [((3, 2), ('3', '4')), ((3, 4), ('3', '4')), ((6, 8), ('6', '16'))]

    we can also randomize the order of the results:

    >>> random.seed(23)
    >>> res = list(FindBRICSBonds(m,randomizeOrder=True))
    >>> sorted(res)
    [((3, 2), ('3', '4')), ((3, 4), ('3', '4')), ((6, 8), ('6', '16'))]

    Note that this is a generator function :

    >>> res = FindBRICSBonds(m)
    >>> res
    <generator object ...>
    >>> next(res)
    ((3, 2), ('3', '4'))

    >>> m = Chem.MolFromSmiles('CC=CC')
    >>> res = list(FindBRICSBonds(m))
    >>> sorted(res)
    [((1, 2), ('7', '7'))]

    make sure we don't match ring bonds:

    >>> m = Chem.MolFromSmiles('O=C1NCCC1')
    >>> list(FindBRICSBonds(m))
    []

    another nice one, make sure environment 8 doesn't match something connected
    to a ring atom:

    >>> m = Chem.MolFromSmiles('CC1(C)CCCCC1')
    >>> list(FindBRICSBonds(m))
    []

    """
    letter = re.compile('[a-z,A-Z]')
    indices = list(range(len(bondMatchers)))
    bondsDone = set()
    if randomizeOrder:
        random.shuffle(indices, random=random.random)
    envMatches = {}
    for env, patt in environMatchers.items():
        envMatches[env] = mol.HasSubstructMatch(patt)
    for gpIdx in indices:
        if randomizeOrder:
            compats = bondMatchers[gpIdx][:]
            random.shuffle(compats, random=random.random)
        else:
            compats = bondMatchers[gpIdx]
        for i1, i2, bType, patt in compats:
            if not envMatches['L' + i1] or not envMatches['L' + i2]:
                continue
            matches = mol.GetSubstructMatches(patt)
            i1 = letter.sub('', i1)
            i2 = letter.sub('', i2)
            for match in matches:
                if match not in bondsDone and (match[1], match[0]) not in bondsDone:
                    bondsDone.add(match)
                    yield ((match[0], match[1]), (i1, i2))