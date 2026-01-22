from warnings import warn
import copy
import logging
from rdkit import Chem
from .utils import memoized_property
class ChargeCorrection(object):
    """An atom that should have a certain charge applied, defined by a SMARTS pattern."""

    def __init__(self, name, smarts, charge):
        """Initialize a ChargeCorrection with the following parameters:

        :param string name: A name for this ForcedAtomCharge.
        :param string smarts: SMARTS pattern to match. Charge is applied to the first atom.
        :param int charge: The charge to apply.
        """
        log.debug(f'Initializing ChargeCorrection: {name}')
        self.name = name
        self.smarts_str = smarts
        self.charge = charge

    @memoized_property
    def smarts(self):
        log.debug(f'Loading ChargeCorrection smarts: {self.name}')
        return Chem.MolFromSmarts(self.smarts_str)

    def __repr__(self):
        return 'ChargeCorrection({!r}, {!r}, {!r})'.format(self.name, self.smarts_str, self.charge)

    def __str__(self):
        return self.name