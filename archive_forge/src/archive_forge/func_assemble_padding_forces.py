import numpy as np
from ase.calculators.calculator import Calculator
from ase.calculators.calculator import compare_atoms
from . import kimpy_wrappers
from . import neighborlist
def assemble_padding_forces(self):
    """
        Assemble forces on padding atoms back to contributing atoms.

        Parameters
        ----------
        forces : 2D array of doubles
            Forces on both contributing and padding atoms

        num_contrib:  int
            Number of contributing atoms

        padding_image_of : 1D array of int
            Atom number, of which the padding atom is an image


        Returns
        -------
            Total forces on contributing atoms.
        """
    total_forces = np.array(self.forces[:self.num_contributing_particles])
    if self.padding_image_of.size != 0:
        pad_forces = self.forces[self.num_contributing_particles:]
        for f, org_index in zip(pad_forces, self.padding_image_of):
            total_forces[org_index] += f
    return total_forces