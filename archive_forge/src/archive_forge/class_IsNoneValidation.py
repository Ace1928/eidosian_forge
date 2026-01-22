from warnings import warn
import logging
from rdkit import Chem
from .errors import StopValidateError
from .fragment import REMOVE_FRAGMENTS
class IsNoneValidation(Validation):
    """Logs an error if ``None`` is passed to the Validator.

    This can happen if RDKit failed to parse an input format. If the molecule is ``None``, no subsequent validations
    will run.
    """

    def run(self, mol):
        if mol is None:
            self.log.error('Molecule is None')
            raise StopValidateError()