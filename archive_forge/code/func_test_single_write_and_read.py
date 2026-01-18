import filecmp
import pytest
from ase.build import molecule
from ase.io import read, write
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
def test_single_write_and_read():
    atoms = molecule('H2O')
    write('1.xyz', atoms, format='extxyz', plain=True)
    write('2.xyz', atoms, format='xyz', fmt='%16.8f')
    assert filecmp.cmp('1.xyz', '2.xyz', shallow=False), 'Files differ'
    atoms1 = read('1.xyz', format='xyz')
    assert atoms_equal(atoms, atoms1), 'Read failed'