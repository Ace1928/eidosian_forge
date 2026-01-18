import re
import numpy as np
import ase
from ase.spacegroup import Spacegroup, crystal
from ase.geometry import cellpar_to_cell, cell_to_cellpar
def write_jsv(fd, atoms):
    """Writes JSV file."""
    fd.write('asymmetric_unit_cell\n')
    fd.write('[cell]')
    for v in cell_to_cellpar(atoms.cell):
        fd.write('  %g' % v)
    fd.write('\n')
    fd.write('[natom]  %d\n' % len(atoms))
    fd.write('[nbond]  0\n')
    fd.write('[npoly]  0\n')
    if 'spacegroup' in atoms.info:
        sg = Spacegroup(atoms.info['spacegroup'])
        fd.write('[space_group]  %d %d\n' % (sg.no, sg.setting))
    else:
        fd.write('[space_group]  1  1\n')
    fd.write('[title] %s\n' % atoms.info.get('title', 'untitled'))
    fd.write('\n')
    fd.write('[atoms]\n')
    if 'labels' in atoms.info:
        labels = atoms.info['labels']
    else:
        labels = ['%s%d' % (s, i + 1) for i, s in enumerate(atoms.get_chemical_symbols())]
    numbers = atoms.get_atomic_numbers()
    scaled = atoms.get_scaled_positions()
    for l, n, p in zip(labels, numbers, scaled):
        fd.write('%-4s  %2d  %9.6f  %9.6f  %9.6f\n' % (l, n, p[0], p[1], p[2]))
    fd.write('Label  AtomicNumber  x y z (repeat natom times)\n')
    fd.write('\n')
    fd.write('[bonds]\n')
    fd.write('\n')
    fd.write('[poly]\n')
    fd.write('\n')