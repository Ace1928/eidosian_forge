import filecmp
import pytest
from ase.build import molecule
from ase.io import read, write
from ase.calculators.calculator import compare_atoms
from ase.calculators.emt import EMT
def test_single_write_with_forces():
    atoms = molecule('CO')
    atoms.calc = EMT()
    atoms.get_forces()
    write('1.xyz', atoms, format='extxyz', plain=True)
    write('2.xyz', atoms, format='xyz', fmt='%16.8f')
    write('3.xyz', molecule('CO'), format='xyz', fmt='%16.8f')
    assert filecmp.cmp('1.xyz', '2.xyz', shallow=False), 'Files differ'
    assert filecmp.cmp('1.xyz', '3.xyz', shallow=False), 'Files differ'