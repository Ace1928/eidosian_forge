from warnings import warn
import copy
import logging
from rdkit import Chem
from .utils import memoized_property
class Uncharger(object):
    """Class for neutralizing ionized acids and bases.

    This class uncharges molecules by adding and/or removing hydrogens. For zwitterions, hydrogens are moved to
    eliminate charges where possible. However, in cases where there is a positive charge that is not neutralizable, an
    attempt is made to also preserve the corresponding negative charge.

    The method is derived from the neutralise module in `Francis Atkinson's standardiser tool
    <https://github.com/flatkinson/standardiser>`_, which is released under the Apache License v2.0.
    """

    def __init__(self):
        log.debug('Initializing Uncharger')
        self._pos_h = Chem.MolFromSmarts('[+!H0!$(*~[-])]')
        self._pos_quat = Chem.MolFromSmarts('[+H0!$(*~[-])]')
        self._neg = Chem.MolFromSmarts('[-!$(*~[+H0])]')
        self._neg_acid = Chem.MolFromSmarts('[$([O-][C,P,S]=O),$([n-]1nnnc1),$(n1[n-]nnc1)]')

    def __call__(self, mol):
        """Calling an Uncharger instance like a function is the same as calling its uncharge(mol) method."""
        return self.uncharge(mol)

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