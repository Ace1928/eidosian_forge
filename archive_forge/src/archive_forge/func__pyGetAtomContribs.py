import os
import numpy
from rdkit import Chem, RDConfig
from rdkit.Chem import rdMolDescriptors
def _pyGetAtomContribs(mol, patts=None, order=None, verbose=0, force=0):
    """ *Internal Use Only*

    calculates atomic contributions to the LogP and MR values

    if the argument *force* is not set, we'll use the molecules stored
    _crippenContribs value when possible instead of re-calculating.

  **Note:** Changes here affect the version numbers of MolLogP and MolMR
    as well as the VSA descriptors in Chem.MolSurf

  """
    if not force and hasattr(mol, '_crippenContribs'):
        return mol._crippenContribs
    if patts is None:
        patts = _smartsPatterns
        order = _patternOrder
    nAtoms = mol.GetNumAtoms()
    atomContribs = [(0.0, 0.0)] * nAtoms
    doneAtoms = [0] * nAtoms
    nAtomsFound = 0
    done = False
    for cha in order:
        pattVect = patts[cha]
        for sma, patt, logp, mr in pattVect:
            for match in mol.GetSubstructMatches(patt, False, False):
                firstIdx = match[0]
                if not doneAtoms[firstIdx]:
                    doneAtoms[firstIdx] = 1
                    atomContribs[firstIdx] = (logp, mr)
                    if verbose:
                        print('\tAtom %d: %s %4.4f %4.4f' % (match[0], sma, logp, mr))
                    nAtomsFound += 1
                    if nAtomsFound >= nAtoms:
                        done = True
                        break
        if done:
            break
    mol._crippenContribs = atomContribs
    return atomContribs