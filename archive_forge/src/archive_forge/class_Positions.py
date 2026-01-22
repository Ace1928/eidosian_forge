import numpy as np
from ase import Atoms
from ase.geometry import find_mic
from ase.ga.utilities import (atoms_too_close, atoms_too_close_two_sets,
from ase.ga.offspring_creator import OffspringCreator
class Positions:
    """Helper object to simplify the pairing process.

    Parameters:

    scaled_positions: (Nx3) array
        Positions in scaled coordinates
    cop: (1x3) array
        Center-of-positions (also in scaled coordinates)
    symbols: str
        String with the atomic symbols
    distance: float
        Signed distance to the cutting plane
    origin: int (0 or 1)
        Determines at which side of the plane the position should be.
    """

    def __init__(self, scaled_positions, cop, symbols, distance, origin):
        self.scaled_positions = scaled_positions
        self.cop = cop
        self.symbols = symbols
        self.distance = distance
        self.origin = origin

    def to_use(self):
        """Tells whether this position is at the right side."""
        if self.distance > 0.0 and self.origin == 0:
            return True
        elif self.distance < 0.0 and self.origin == 1:
            return True
        else:
            return False