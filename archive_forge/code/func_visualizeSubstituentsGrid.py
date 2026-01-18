import re
from collections import Counter, defaultdict, namedtuple
import numpy as np
import seaborn as sns
from numpy import linalg
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem.Draw import rdMolDraw2D
def visualizeSubstituentsGrid(mol, aIdx, molSize=(300, 150), kekulize=True):
    dists = Chem.GetDistanceMatrix(mol)
    idxChiral = Chem.FindMolChiralCenters(mol)[0][0]
    subs, sharedNeighbors, maxShell = determineAtomSubstituents(aIdx, mol, dists, False)
    colors = sns.husl_palette(len(subs), s=0.6)
    mc = rdMolDraw2D.PrepareMolForDrawing(mol, kekulize=kekulize)
    count = 0
    svgs = []
    labels = []
    for sub in sorted(subs.values(), key=lambda x: _getSizeOfSubstituents(x, sharedNeighbors)):
        color = tuple(colors[count])
        count += 1
        atColors = {atom: color for atom in sub}
        bonds = getBondsSubstituent(mol, set(sub))
        bnColors = {bond: color for bond in bonds}
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0], molSize[1])
        drawer.DrawMolecule(mc, highlightAtoms=atColors.keys(), highlightAtomColors=atColors, highlightBonds=bonds, highlightBondColors=bnColors)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        svgs.append(svg.replace('svg:', ''))
        labels.append('Substituent ' + str(count) + ' (#atoms: ' + str(len(sub)) + ', size normed: ' + str(_getSizeOfSubstituents(sub, sharedNeighbors)) + ')')
    return _svgsToGrid(svgs, labels, svgsPerRow=len(svgs), molSize=molSize, fontSize=12)