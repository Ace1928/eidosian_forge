from math import sqrt
from ase.atoms import Atoms
from ase.symbols import string2symbols
from ase.data import reference_states, atomic_numbers, chemical_symbols
from ase.utils import plural
def _orthorhombic_bulk(name, crystalstructure, a, covera=None, u=None):
    if crystalstructure == 'fcc':
        b = a / sqrt(2)
        atoms = Atoms(2 * name, cell=(b, b, a), pbc=True, scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)])
    elif crystalstructure == 'bcc':
        atoms = Atoms(2 * name, cell=(a, a, a), pbc=True, scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)])
    elif crystalstructure == 'hcp':
        atoms = Atoms(4 * name, cell=(a, a * sqrt(3), covera * a), scaled_positions=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 1 / 6, 0.5), (0, 2 / 3, 0.5)], pbc=True)
    elif crystalstructure == 'diamond':
        atoms = _orthorhombic_bulk(2 * name, 'zincblende', a)
    elif crystalstructure == 'zincblende':
        s1, s2 = string2symbols(name)
        b = a / sqrt(2)
        atoms = Atoms(2 * name, cell=(b, b, a), pbc=True, scaled_positions=[(0, 0, 0), (0.5, 0, 0.25), (0.5, 0.5, 0.5), (0, 0.5, 0.75)])
    elif crystalstructure == 'rocksalt':
        s1, s2 = string2symbols(name)
        b = a / sqrt(2)
        atoms = Atoms(2 * name, cell=(b, b, a), pbc=True, scaled_positions=[(0, 0, 0), (0.5, 0.5, 0), (0.5, 0.5, 0.5), (0, 0, 0.5)])
    elif crystalstructure == 'cesiumchloride':
        atoms = Atoms(name, cell=(a, a, a), pbc=True, scaled_positions=[(0, 0, 0), (0.5, 0.5, 0.5)])
    elif crystalstructure == 'wurtzite':
        u = u or 0.25 + 1 / 3 / covera ** 2
        atoms = Atoms(4 * name, cell=(a, a * 3 ** 0.5, covera * a), scaled_positions=[(0, 0, 0), (0, 1 / 3, 0.5 - u), (0, 1 / 3, 0.5), (0, 0, 1 - u), (0.5, 0.5, 0), (0.5, 5 / 6, 0.5 - u), (0.5, 5 / 6, 0.5), (0.5, 0.5, 1 - u)], pbc=True)
    else:
        raise incompatible_cell(want='orthorhombic', have=crystalstructure)
    return atoms