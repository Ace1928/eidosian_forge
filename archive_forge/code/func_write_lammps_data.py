import re
import numpy as np
from ase.atoms import Atoms
from ase.calculators.lammps import Prism, convert
from ase.utils import reader, writer
@writer
def write_lammps_data(fd, atoms, specorder=None, force_skew=False, prismobj=None, velocities=False, units='metal', atom_style='atomic'):
    """Write atomic structure data to a LAMMPS data file."""
    if isinstance(atoms, list):
        if len(atoms) > 1:
            raise ValueError('Can only write one configuration to a lammps data file!')
        atoms = atoms[0]
    if hasattr(fd, 'name'):
        fd.write('{0} (written by ASE) \n\n'.format(fd.name))
    else:
        fd.write('(written by ASE) \n\n')
    symbols = atoms.get_chemical_symbols()
    n_atoms = len(symbols)
    fd.write('{0} \t atoms \n'.format(n_atoms))
    if specorder is None:
        species = sorted(set(symbols))
    else:
        species = specorder
    n_atom_types = len(species)
    fd.write('{0}  atom types\n'.format(n_atom_types))
    if prismobj is None:
        p = Prism(atoms.get_cell())
    else:
        p = prismobj
    xhi, yhi, zhi, xy, xz, yz = convert(p.get_lammps_prism(), 'distance', 'ASE', units)
    fd.write('0.0 {0:23.17g}  xlo xhi\n'.format(xhi))
    fd.write('0.0 {0:23.17g}  ylo yhi\n'.format(yhi))
    fd.write('0.0 {0:23.17g}  zlo zhi\n'.format(zhi))
    if force_skew or p.is_skewed():
        fd.write('{0:23.17g} {1:23.17g} {2:23.17g}  xy xz yz\n'.format(xy, xz, yz))
    fd.write('\n\n')
    fd.write('Atoms \n\n')
    pos = p.vector_to_lammps(atoms.get_positions(), wrap=False)
    if atom_style == 'atomic':
        for i, r in enumerate(pos):
            r = convert(r, 'distance', 'ASE', units)
            s = species.index(symbols[i]) + 1
            fd.write('{0:>6} {1:>3} {2:23.17g} {3:23.17g} {4:23.17g}\n'.format(*(i + 1, s) + tuple(r)))
    elif atom_style == 'charge':
        charges = atoms.get_initial_charges()
        for i, (q, r) in enumerate(zip(charges, pos)):
            r = convert(r, 'distance', 'ASE', units)
            q = convert(q, 'charge', 'ASE', units)
            s = species.index(symbols[i]) + 1
            fd.write('{0:>6} {1:>3} {2:>5} {3:23.17g} {4:23.17g} {5:23.17g}\n'.format(*(i + 1, s, q) + tuple(r)))
    elif atom_style == 'full':
        charges = atoms.get_initial_charges()
        if atoms.has('mol-id'):
            molecules = atoms.get_array('mol-id')
            if not np.issubdtype(molecules.dtype, np.integer):
                raise TypeError("If 'atoms' object has 'mol-id' array, then mol-id dtype must be subtype of np.integer, and not {:s}.".format(str(molecules.dtype)))
            if len(molecules) != len(atoms) or molecules.ndim != 1:
                raise TypeError("If 'atoms' object has 'mol-id' array, then each atom must have exactly one mol-id.")
        else:
            molecules = np.zeros(len(atoms), dtype=int)
        for i, (m, q, r) in enumerate(zip(molecules, charges, pos)):
            r = convert(r, 'distance', 'ASE', units)
            q = convert(q, 'charge', 'ASE', units)
            s = species.index(symbols[i]) + 1
            fd.write('{0:>6} {1:>3} {2:>3} {3:>5} {4:23.17g} {5:23.17g} {6:23.17g}\n'.format(*(i + 1, m, s, q) + tuple(r)))
    else:
        raise NotImplementedError
    if velocities and atoms.get_velocities() is not None:
        fd.write('\n\nVelocities \n\n')
        vel = p.vector_to_lammps(atoms.get_velocities())
        for i, v in enumerate(vel):
            v = convert(v, 'velocity', 'ASE', units)
            fd.write('{0:>6} {1:23.17g} {2:23.17g} {3:23.17g}\n'.format(*(i + 1,) + tuple(v)))
    fd.flush()