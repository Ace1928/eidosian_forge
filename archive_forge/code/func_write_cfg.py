import numpy as np
import ase
from ase.data import chemical_symbols
from ase.utils import reader, writer
@writer
def write_cfg(fd, atoms):
    """Write atomic configuration to a CFG-file (native AtomEye format).
       See: http://mt.seas.upenn.edu/Archive/Graphics/A/
    """
    fd.write('Number of particles = %i\n' % len(atoms))
    fd.write('A = 1.0 Angstrom\n')
    cell = atoms.get_cell(complete=True)
    for i in range(3):
        for j in range(3):
            fd.write('H0(%1.1i,%1.1i) = %f A\n' % (i + 1, j + 1, cell[i, j]))
    entry_count = 3
    for x in atoms.arrays.keys():
        if x not in cfg_default_fields:
            if len(atoms.get_array(x).shape) == 1:
                entry_count += 1
            else:
                entry_count += atoms.get_array(x).shape[1]
    vels = atoms.get_velocities()
    if isinstance(vels, np.ndarray):
        entry_count += 3
    else:
        fd.write('.NO_VELOCITY.\n')
    fd.write('entry_count = %i\n' % entry_count)
    i = 0
    for name, aux in atoms.arrays.items():
        if name not in cfg_default_fields:
            if len(aux.shape) == 1:
                fd.write('auxiliary[%i] = %s [a.u.]\n' % (i, name))
                i += 1
            elif aux.shape[1] == 3:
                for j in range(3):
                    fd.write('auxiliary[%i] = %s_%s [a.u.]\n' % (i, name, chr(ord('x') + j)))
                    i += 1
            else:
                for j in range(aux.shape[1]):
                    fd.write('auxiliary[%i] = %s_%1.1i [a.u.]\n' % (i, name, j))
                    i += 1
    spos = atoms.get_scaled_positions()
    for i in atoms:
        el = i.symbol
        fd.write('%f\n' % ase.data.atomic_masses[chemical_symbols.index(el)])
        fd.write('%s\n' % el)
        x, y, z = spos[i.index, :]
        s = '%e %e %e ' % (x, y, z)
        if isinstance(vels, np.ndarray):
            vx, vy, vz = vels[i.index, :]
            s = s + ' %e %e %e ' % (vx, vy, vz)
        for name, aux in atoms.arrays.items():
            if name not in cfg_default_fields:
                if len(aux.shape) == 1:
                    s += ' %e' % aux[i.index]
                else:
                    s += aux.shape[1] * ' %e' % tuple(aux[i.index].tolist())
        fd.write('%s\n' % s)