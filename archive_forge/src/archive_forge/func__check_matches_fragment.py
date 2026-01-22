from warnings import warn
import logging
from rdkit import Chem
from .errors import StopValidateError
from .fragment import REMOVE_FRAGMENTS
def _check_matches_fragment(self, mol):
    matches = frozenset((frozenset(match) for match in mol.GetSubstructMatches(self._smarts)))
    fragments = frozenset((frozenset(frag) for frag in Chem.GetMolFrags(mol)))
    if matches & fragments:
        self.log.log(self.level, self.message, {'smarts': self.smarts})