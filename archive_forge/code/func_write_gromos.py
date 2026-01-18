import numpy as np
from ase import Atoms
from ase.data import chemical_symbols
from ase.utils import reader, writer
@writer
def write_gromos(fileobj, atoms):
    """Write gromos geometry files (.g96).
    Writes:
    atom positions,
    and simulation cell (if present)
    """
    from ase import units
    natoms = len(atoms)
    try:
        gromos_residuenames = atoms.get_array('residuenames')
    except KeyError:
        gromos_residuenames = []
        for idum in range(natoms):
            gromos_residuenames.append('1DUM')
    try:
        gromos_atomtypes = atoms.get_array('atomtypes')
    except KeyError:
        gromos_atomtypes = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    pos = pos / 10.0
    vel = atoms.get_velocities()
    if vel is None:
        vel = pos * 0.0
    else:
        vel *= 1000.0 * units.fs / units.nm
    fileobj.write('TITLE\n')
    fileobj.write('Gromos96 structure file written by ASE \n')
    fileobj.write('END\n')
    fileobj.write('POSITION\n')
    count = 1
    rescount = 0
    oldresname = ''
    for resname, atomtype, xyz in zip(gromos_residuenames, gromos_atomtypes, pos):
        if resname != oldresname:
            oldresname = resname
            rescount = rescount + 1
        okresname = resname.lstrip('0123456789 ')
        fileobj.write('%5d %-5s %-5s%7d%15.9f%15.9f%15.9f\n' % (rescount, okresname, atomtype, count, xyz[0], xyz[1], xyz[2]))
        count = count + 1
    fileobj.write('END\n')
    if atoms.get_pbc().any():
        fileobj.write('BOX\n')
        mycell = atoms.get_cell()
        grocell = mycell.flat[[0, 4, 8, 1, 2, 3, 5, 6, 7]] * 0.1
        fileobj.write(''.join(['{:15.9f}'.format(x) for x in grocell]))
        fileobj.write('\nEND\n')