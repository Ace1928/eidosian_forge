import argparse
import sys
from rdkit import Chem, Geometry
from rdkit.Chem import rdDepictor
def AlignDepict(mol, core, corePattern=None, acceptFailure=False):
    """
  Arguments:
    - mol:          the molecule to be aligned, this will come back
                    with a single conformer.
    - core:         a molecule with the core atoms to align to;
                    this should have a depiction.
    - corePattern:  (optional) an optional molecule to be used to
                    generate the atom mapping between the molecule
                    and the core.
  """
    if core and corePattern:
        if not core.GetNumAtoms(onlyExplicit=True) == corePattern.GetNumAtoms(onlyExplicit=True):
            raise ValueError('When a pattern is provided, it must have the same number of atoms as the core')
        coreMatch = core.GetSubstructMatch(corePattern)
        if not coreMatch:
            raise ValueError('Core does not map to itself')
    else:
        coreMatch = list(range(core.GetNumAtoms(onlyExplicit=True)))
    if corePattern:
        match = mol.GetSubstructMatch(corePattern)
    else:
        match = mol.GetSubstructMatch(core)
    if not match:
        if not acceptFailure:
            raise ValueError('Substructure match with core not found.')
        else:
            coordMap = {}
    else:
        conf = core.GetConformer()
        coordMap = {}
        for i, idx in enumerate(match):
            pt3 = conf.GetAtomPosition(coreMatch[i])
            coordMap[idx] = Geometry.Point2D(pt3.x, pt3.y)
    rdDepictor.Compute2DCoords(mol, clearConfs=True, coordMap=coordMap, canonOrient=False)