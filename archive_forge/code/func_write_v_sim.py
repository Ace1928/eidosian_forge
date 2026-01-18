import numpy as np
from ase.utils import reader, writer
@writer
def write_v_sim(fd, atoms):
    """Write V_Sim input file.

    Writes the atom positions and unit cell.
    """
    from ase.geometry import cellpar_to_cell, cell_to_cellpar
    cell = cellpar_to_cell(cell_to_cellpar(atoms.cell))
    dxx = cell[0, 0]
    dyx, dyy = cell[1, 0:2]
    dzx, dzy, dzz = cell[2, 0:3]
    fd.write('===== v_sim input file created using the Atomic Simulation Environment (ASE) ====\n')
    fd.write('{0} {1} {2}\n'.format(dxx, dyx, dyy))
    fd.write('{0} {1} {2}\n'.format(dzx, dzy, dzz))
    fd.write('#keyword: reduced\n')
    fd.write('#keyword: angstroem\n')
    if np.alltrue(atoms.pbc):
        fd.write('#keyword: periodic\n')
    elif not np.any(atoms.pbc):
        fd.write('#keyword: freeBC\n')
    elif np.array_equiv(atoms.pbc, [True, False, True]):
        fd.write('#keyword: surface\n')
    else:
        raise Exception('Only supported boundary conditions are full PBC, no periodic boundary, and surface which is free in y direction (i.e. Atoms.pbc = [True, False, True]).')
    for position, symbol in zip(atoms.get_scaled_positions(), atoms.get_chemical_symbols()):
        fd.write('{0} {1} {2} {3}\n'.format(position[0], position[1], position[2], symbol))