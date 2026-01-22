import numpy as np
from ase import Atoms
class SupercellError(Exception):
    """Use if construction of supercell fails"""