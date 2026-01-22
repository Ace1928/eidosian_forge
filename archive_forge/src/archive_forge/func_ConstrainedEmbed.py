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
def ConstrainedEmbed(mol, core, useTethers=True, coreConfId=-1, randomseed=2342, getForceField=UFFGetMoleculeForceField, **kwargs):
    """ generates an embedding of a molecule where part of the molecule
    is constrained to have particular coordinates

    Arguments
      - mol: the molecule to embed
      - core: the molecule to use as a source of constraints
      - useTethers: (optional) if True, the final conformation will be
          optimized subject to a series of extra forces that pull the
          matching atoms to the positions of the core atoms. Otherwise
          simple distance constraints based on the core atoms will be
          used in the optimization.
      - coreConfId: (optional) id of the core conformation to use
      - randomSeed: (optional) seed for the random number generator


    An example, start by generating a template with a 3D structure:

    >>> from rdkit.Chem import AllChem
    >>> template = AllChem.MolFromSmiles("c1nn(Cc2ccccc2)cc1")
    >>> AllChem.EmbedMolecule(template)
    0
    >>> AllChem.UFFOptimizeMolecule(template)
    0

    Here's a molecule:

    >>> mol = AllChem.MolFromSmiles("c1nn(Cc2ccccc2)cc1-c3ccccc3")

    Now do the constrained embedding
  
    >>> mol = AllChem.ConstrainedEmbed(mol, template)

    Demonstrate that the positions are nearly the same with template:

    >>> import math
    >>> molp = mol.GetConformer().GetAtomPosition(0)
    >>> templatep = template.GetConformer().GetAtomPosition(0)
    >>> all(math.isclose(v, 0.0, abs_tol=0.01) for v in molp-templatep)
    True
    >>> molp = mol.GetConformer().GetAtomPosition(1)
    >>> templatep = template.GetConformer().GetAtomPosition(1)
    >>> all(math.isclose(v, 0.0, abs_tol=0.01) for v in molp-templatep)
    True

    """
    match = mol.GetSubstructMatch(core)
    if not match:
        raise ValueError("molecule doesn't match the core")
    coordMap = {}
    coreConf = core.GetConformer(coreConfId)
    for i, idxI in enumerate(match):
        corePtI = coreConf.GetAtomPosition(i)
        coordMap[idxI] = corePtI
    ci = EmbedMolecule(mol, coordMap=coordMap, randomSeed=randomseed, **kwargs)
    if ci < 0:
        raise ValueError('Could not embed molecule.')
    algMap = [(j, i) for i, j in enumerate(match)]
    if not useTethers:
        ff = getForceField(mol, confId=0)
        for i, idxI in enumerate(match):
            for j in range(i + 1, len(match)):
                idxJ = match[j]
                d = coordMap[idxI].Distance(coordMap[idxJ])
                ff.AddDistanceConstraint(idxI, idxJ, d, d, 100.0)
        ff.Initialize()
        n = 4
        more = ff.Minimize()
        while more and n:
            more = ff.Minimize()
            n -= 1
        rms = AlignMol(mol, core, atomMap=algMap)
    else:
        rms = AlignMol(mol, core, atomMap=algMap)
        ff = getForceField(mol, confId=0)
        conf = core.GetConformer()
        for i in range(core.GetNumAtoms()):
            p = conf.GetAtomPosition(i)
            pIdx = ff.AddExtraPoint(p.x, p.y, p.z, fixed=True) - 1
            ff.AddDistanceConstraint(pIdx, match[i], 0, 0, 100.0)
        ff.Initialize()
        n = 4
        more = ff.Minimize(energyTol=0.0001, forceTol=0.001)
        while more and n:
            more = ff.Minimize(energyTol=0.0001, forceTol=0.001)
            n -= 1
        rms = AlignMol(mol, core, atomMap=algMap)
    mol.SetProp('EmbedRMS', str(rms))
    return mol