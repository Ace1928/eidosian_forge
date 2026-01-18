from the ones used in ase).  Mapping is therefore necessary.
from ase import units
from . import unitconvert_constants as u
Convert units between LAMMPS and ASE.

    :param value: converted value
    :param quantity: mass, distance, time, energy, velocity, force, torque,
    temperature, pressure, dynamic_viscosity, charge, dipole,
    electric_field or density
    :param fromunits: ASE, metal, real or other (see lammps docs).
    :param tounits: ASE, metal, real or other
    :returns: converted value
    :rtype:
    