from warnings import warn
import logging
import sys
from rdkit import Chem
from .errors import StopValidateError
from .validations import VALIDATIONS
def validate_smiles(smiles):
    """Return log messages for a given SMILES string using the default validations.

    Note: This is a convenience function for quickly validating a single SMILES string. It is more efficient to use
    the :class:`~molvs.validate.Validator` class directly when working with many molecules or when custom options
    are needed.

    :param string smiles: The SMILES for the molecule.
    :returns: A list of log messages.
    :rtype: list of strings.
    """
    warn(f'The function validate_smiles is deprecated and will be removed in the next release.', DeprecationWarning, stacklevel=2)
    mol = Chem.MolFromSmiles(smiles)
    logs = Validator().validate(mol)
    return logs