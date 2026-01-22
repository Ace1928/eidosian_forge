import numpy as np
from ase.parallel import world
from ase import units
from ase.md.md import process_temperature
class UnitError(Exception):
    """Exception raised when wrong units are specified"""