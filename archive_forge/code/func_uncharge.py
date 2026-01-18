from warnings import warn
import copy
import logging
from rdkit import Chem
from .utils import memoized_property
def uncharge(self, mol):
    """Neutralize molecule by adding/removing hydrogens. Attempts to preserve zwitterions.

        :param mol: The molecule to uncharge.
        :type mol: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        :return: The uncharged molecule.
        :rtype: :rdkit:`Mol <Chem.rdchem.Mol-class.html>`
        """
    log.debug('Running Uncharger')
    mol = copy.deepcopy(mol)
    p = [x[0] for x in mol.GetSubstructMatches(self._pos_h)]
    q = [x[0] for x in mol.GetSubstructMatches(self._pos_quat)]
    n = [x[0] for x in mol.GetSubstructMatches(self._neg)]
    a = [x[0] for x in mol.GetSubstructMatches(self._neg_acid)]
    if q:
        neg_surplus = len(n) - len(q)
        if a and neg_surplus > 0:
            while neg_surplus > 0 and a:
                atom = mol.GetAtomWithIdx(a.pop(0))
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
                atom.SetFormalCharge(atom.GetFormalCharge() + 1)
                neg_surplus -= 1
                log.info('Removed negative charge')
    else:
        for atom in [mol.GetAtomWithIdx(x) for x in n]:
            while atom.GetFormalCharge() < 0:
                atom.SetNumExplicitHs(atom.GetNumExplicitHs() + 1)
                atom.SetFormalCharge(atom.GetFormalCharge() + 1)
                log.info('Removed negative charge')
    for atom in [mol.GetAtomWithIdx(x) for x in p]:
        while atom.GetFormalCharge() > 0 and atom.GetNumExplicitHs() > 0:
            atom.SetNumExplicitHs(atom.GetNumExplicitHs() - 1)
            atom.SetFormalCharge(atom.GetFormalCharge() - 1)
            log.info('Removed positive charge')
    return mol