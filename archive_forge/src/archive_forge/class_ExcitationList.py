import numpy as np
from ase.units import Hartree, Bohr
class ExcitationList(list):
    """Base class for excitions from the ground state"""

    def __init__(self):
        super().__init__()
        self.energy_to_eV_scale = 1.0