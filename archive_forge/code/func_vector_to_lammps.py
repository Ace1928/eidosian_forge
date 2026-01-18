import numpy as np
from ase.geometry import wrap_positions
def vector_to_lammps(self, vec, wrap=False):
    """Rotate vector from ase coordinate system to lammps one

        :param vec: to be rotated ase-vector
        :returns: lammps-vector
        :rtype: np.array

        """
    if wrap:
        return wrap_positions(np.dot(vec, self.rot_mat), self.lammps_cell, pbc=self.pbc, eps=1e-18)
    return np.dot(vec, self.rot_mat)