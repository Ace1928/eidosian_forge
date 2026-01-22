from math import sqrt
from warnings import warn
import numpy as np
from scipy.linalg import expm, logm
from ase.calculators.calculator import PropertyNotImplementedError
from ase.geometry import (find_mic, wrap_positions, get_distances_derivatives,
from ase.utils.parsemath import eval_expression
from ase.stress import (full_3x3_to_voigt_6_stress,
class StrainFilter(Filter):
    """Modify the supercell while keeping the scaled positions fixed.

    Presents the strain of the supercell as the generalized positions,
    and the global stress tensor (times the volume) as the generalized
    force.

    This filter can be used to relax the unit cell until the stress is
    zero.  If MDMin is used for this, the timestep (dt) to be used
    depends on the system size. 0.01/x where x is a typical dimension
    seems like a good choice.

    The stress and strain are presented as 6-vectors, the order of the
    components follow the standard engingeering practice: xx, yy, zz,
    yz, xz, xy.

    """

    def __init__(self, atoms, mask=None, include_ideal_gas=False):
        """Create a filter applying a homogeneous strain to a list of atoms.

        The first argument, atoms, is the atoms object.

        The optional second argument, mask, is a list of six booleans,
        indicating which of the six independent components of the
        strain that are allowed to become non-zero.  It defaults to
        [1,1,1,1,1,1].

        """
        self.strain = np.zeros(6)
        self.include_ideal_gas = include_ideal_gas
        if mask is None:
            mask = np.ones(6)
        else:
            mask = np.array(mask)
        Filter.__init__(self, atoms, mask=mask)
        self.mask = mask
        self.origcell = atoms.get_cell()

    def get_positions(self):
        return self.strain.reshape((2, 3)).copy()

    def set_positions(self, new):
        new = new.ravel() * self.mask
        eps = np.array([[1.0 + new[0], 0.5 * new[5], 0.5 * new[4]], [0.5 * new[5], 1.0 + new[1], 0.5 * new[3]], [0.5 * new[4], 0.5 * new[3], 1.0 + new[2]]])
        self.atoms.set_cell(np.dot(self.origcell, eps), scale_atoms=True)
        self.strain[:] = new

    def get_forces(self, **kwargs):
        stress = self.atoms.get_stress(include_ideal_gas=self.include_ideal_gas)
        return -self.atoms.get_volume() * (stress * self.mask).reshape((2, 3))

    def has(self, x):
        return self.atoms.has(x)

    def __len__(self):
        return 2