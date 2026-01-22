from math import sqrt
from ase.atoms import Atoms
from ase.symbols import string2symbols
from ase.data import reference_states, atomic_numbers, chemical_symbols
from ase.utils import plural
def _build_rhl(name, a, alpha, basis):
    from ase.lattice import RHL
    lat = RHL(a, alpha)
    cell = lat.tocell()
    if basis is None:
        basis_x = reference_states[atomic_numbers[name]]['basis_x']
        basis = basis_x[:, None].repeat(3, axis=1)
    natoms = len(basis)
    return Atoms([name] * natoms, cell=cell, scaled_positions=basis, pbc=True)