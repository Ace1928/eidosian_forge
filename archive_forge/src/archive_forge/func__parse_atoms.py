import time
import warnings
from ase.units import Ang, fs
from ase.utils import reader, writer
def _parse_atoms(fd, n_atoms, molecular_dynamics=False):
    """parse structure information from aims output to Atoms object"""
    from ase import Atoms, Atom
    next(fd)
    atoms = Atoms()
    for i in range(n_atoms):
        inp = next(fd).split()
        if 'lattice_vector' in inp[0]:
            cell = []
            for i in range(3):
                cell += [[float(inp[1]), float(inp[2]), float(inp[3])]]
                inp = next(fd).split()
            atoms.set_cell(cell)
            inp = next(fd).split()
        atoms.append(Atom(inp[4], (inp[1], inp[2], inp[3])))
        if molecular_dynamics:
            inp = next(fd).split()
    return atoms