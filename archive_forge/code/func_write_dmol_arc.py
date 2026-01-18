from datetime import datetime
import numpy as np
from ase import Atom, Atoms
from ase.geometry.cell import cell_to_cellpar, cellpar_to_cell
from ase.units import Bohr
from ase.utils import reader, writer
@writer
def write_dmol_arc(fd, images):
    """ Writes all images to file filename in arc format.

    Similar to the .car format only pbc 111 or 000 is supported.
    """
    fd.write('!BIOSYM archive 3\n')
    if np.all(images[0].pbc):
        fd.write('PBC=ON\n\n')
    elif not np.any(images[0].pbc):
        fd.write('PBC=OFF\n\n')
    else:
        raise ValueError('PBC must be all true or all false for .arc format')
    for atoms in images:
        dt = datetime.now()
        symbols = atoms.get_chemical_symbols()
        if np.all(atoms.pbc):
            cellpar = cell_to_cellpar(atoms.cell)
            new_cell = cellpar_to_cell(cellpar)
            lstsq_fit = np.linalg.lstsq(atoms.cell, new_cell, rcond=-1)
            R = lstsq_fit[0]
            fd.write('!DATE     %s\n' % dt.strftime('%b %d %H:%m:%S %Y'))
            fd.write('PBC %9.5f %9.5f %9.5f %9.5f %9.5f %9.5f\n' % tuple(cellpar))
            positions = np.dot(atoms.positions, R)
        elif not np.any(atoms.pbc):
            fd.write('!DATE     %s\n' % dt.strftime('%b %d %H:%m:%S %Y'))
            positions = atoms.positions
        else:
            raise ValueError('PBC must be all true or all false for .arc format')
        for i, (sym, pos) in enumerate(zip(symbols, positions)):
            fd.write('%-6s  %12.8f   %12.8f   %12.8f XXXX 1      xx      %-2s  0.000\n' % (sym + str(i + 1), pos[0], pos[1], pos[2], sym))
        fd.write('end\nend\n')
        fd.write('\n')