import numpy as np
from ase import io
from ase.build import bulk
from ase.io.gpumd import load_xyz_input_gpumd
def test_read_gpumd_input():
    """Read GPUMD input file."""
    with open('xyz.in', 'w') as fd:
        fd.write(gpumd_input_text)
    species = ['Si', 'C']
    atoms = io.read('xyz.in', format='gpumd', species=species)
    groupings = [[[i] for i in range(len(atoms))], [[i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'Si'], [i for i, s in enumerate(atoms.get_chemical_symbols()) if s == 'C']]]
    groups = [[[j for j, group in enumerate(grouping) if i in group][0] for grouping in groupings] for i in range(len(atoms))]
    assert len(atoms) == 16
    assert set(atoms.symbols) == set(species)
    assert all(atoms.get_pbc())
    assert len(atoms.info) == len(atoms)
    assert all((np.array_equal(atoms.info[i]['groups'], np.array(groups[i])) for i in range(len(atoms))))
    assert len(atoms.get_velocities()) == len(atoms)
    atoms = io.read('xyz.in', format='gpumd')
    assert set(atoms.symbols) == set(species)
    isotope_masses = {'Si': [28.085], 'C': [12.011]}
    atoms = io.read('xyz.in', format='gpumd', isotope_masses=isotope_masses)
    assert set(atoms.symbols) == set(species)