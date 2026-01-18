import numpy as np
from ase.atoms import Atoms
from ase.utils import reader, writer
def write_dftb_velocities(atoms, filename):
    """Method to write velocities (in atomic units) from ASE
       to a file to be read by dftb+
    """
    from ase.units import AUT, Bohr
    ASE2au = Bohr / AUT
    with open(filename, 'w') as fd:
        velocities = atoms.get_velocities()
        for velocity in velocities:
            fd.write(' %19.16f %19.16f %19.16f \n' % (velocity[0] / ASE2au, velocity[1] / ASE2au, velocity[2] / ASE2au))