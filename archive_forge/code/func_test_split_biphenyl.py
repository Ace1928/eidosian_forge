from ase import Atoms
from ase.build import molecule
from ase.build.connected import connected_atoms, split_bond, separate
from ase.data.s22 import data
def test_split_biphenyl():
    mol = molecule('biphenyl')
    mol1, mol2 = split_bond(mol, 0, 14)
    assert len(mol) == len(mol1) + len(mol2)
    mol2s, mol1s = split_bond(mol, 14, 0)
    assert mol1s == mol1
    assert mol2s == mol2
    mol1, mol2 = split_bond(mol, 0, 1)
    assert len(mol) < len(mol1) + len(mol2)