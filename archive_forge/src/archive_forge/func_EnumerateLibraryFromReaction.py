import sys
import warnings
from collections import namedtuple
import numpy
from rdkit import DataStructs, ForceField, RDConfig, rdBase
from rdkit.Chem import *
from rdkit.Chem.ChemicalFeatures import *
from rdkit.Chem.EnumerateStereoisomers import (EnumerateStereoisomers,
from rdkit.Chem.rdChemReactions import *
from rdkit.Chem.rdDepictor import *
from rdkit.Chem.rdDistGeom import *
from rdkit.Chem.rdFingerprintGenerator import *
from rdkit.Chem.rdForceFieldHelpers import *
from rdkit.Chem.rdMolAlign import *
from rdkit.Chem.rdMolDescriptors import *
from rdkit.Chem.rdMolEnumerator import *
from rdkit.Chem.rdMolTransforms import *
from rdkit.Chem.rdPartialCharges import *
from rdkit.Chem.rdqueries import *
from rdkit.Chem.rdReducedGraphs import *
from rdkit.Chem.rdShapeHelpers import *
from rdkit.Geometry import rdGeometry
from rdkit.RDLogger import logger
def EnumerateLibraryFromReaction(reaction, sidechainSets, returnReactants=False):
    """ Returns a generator for the virtual library defined by
    a reaction and a sequence of sidechain sets

    >>> from rdkit import Chem
    >>> from rdkit.Chem import AllChem
    >>> s1=[Chem.MolFromSmiles(x) for x in ('NC','NCC')]
    >>> s2=[Chem.MolFromSmiles(x) for x in ('OC=O','OC(=O)C')]
    >>> rxn = AllChem.ReactionFromSmarts('[O:2]=[C:1][OH].[N:3]>>[O:2]=[C:1][N:3]')
    >>> r = AllChem.EnumerateLibraryFromReaction(rxn,[s2,s1])
    >>> [Chem.MolToSmiles(x[0]) for x in list(r)]
    ['CNC=O', 'CCNC=O', 'CNC(C)=O', 'CCNC(C)=O']

    Note that this is all done in a lazy manner, so "infinitely" large libraries can
    be done without worrying about running out of memory. Your patience will run out first:

    Define a set of 10000 amines:

    >>> amines = (Chem.MolFromSmiles('N'+'C'*x) for x in range(10000))

    ... a set of 10000 acids

    >>> acids = (Chem.MolFromSmiles('OC(=O)'+'C'*x) for x in range(10000))

    ... now the virtual library (1e8 compounds in principle):

    >>> r = AllChem.EnumerateLibraryFromReaction(rxn,[acids,amines])

    ... look at the first 4 compounds:

    >>> [Chem.MolToSmiles(next(r)[0]) for x in range(4)]
    ['NC=O', 'CNC=O', 'CCNC=O', 'CCCNC=O']


    """
    if len(sidechainSets) != reaction.GetNumReactantTemplates():
        raise ValueError('%d sidechains provided, %d required' % (len(sidechainSets), reaction.GetNumReactantTemplates()))

    def _combiEnumerator(items, depth=0):
        for item in items[depth]:
            if depth + 1 < len(items):
                v = _combiEnumerator(items, depth + 1)
                for entry in v:
                    l = [item]
                    l.extend(entry)
                    yield l
            else:
                yield [item]
    ProductReactants = namedtuple('ProductReactants', 'products,reactants')
    for chains in _combiEnumerator(sidechainSets):
        prodSets = reaction.RunReactants(chains)
        for prods in prodSets:
            if returnReactants:
                yield ProductReactants(prods, chains)
            else:
                yield prods