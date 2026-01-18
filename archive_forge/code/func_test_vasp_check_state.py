import os
import pytest
@calc('vasp')
def test_vasp_check_state(factory, atoms_2co):
    """
    Run tests to ensure that the VASP check_state() function call works correctly,
    i.e. correctly sets the working directories and works in that directory.

    This is conditional on the existence of the VASP_COMMAND or VASP_SCRIPT
    environment variables

    """
    atoms = atoms_2co
    settings = dict(xc='LDA', prec='Low', algo='Fast', ismear=0, sigma=1.0, istart=0, lwave=False, lcharg=False)
    s1 = atoms.get_chemical_symbols()
    calc = factory.calc(**settings)
    atoms.calc = calc
    en1 = atoms.get_potential_energy()
    fi = 'json_test.json'
    calc.write_json(filename=fi)
    assert os.path.isfile(fi)
    calc2 = factory.calc()
    calc2.read_json(fi)
    assert not calc2.calculation_required(atoms, ['energy', 'forces'])
    en2 = calc2.get_potential_energy()
    assert abs(en1 - en2) < 1e-08
    os.remove(fi)
    s2 = calc.atoms.get_chemical_symbols()
    assert s1 == s2
    s3 = sorted(s2)
    assert s2 != s3
    r1 = dict(calc.results)
    calc.get_atoms()
    r2 = dict(calc.results)
    assert r1 == r2
    calc.set(sigma=0.5)
    assert calc.check_state(atoms) == ['float_params']
    assert calc.calculation_required(atoms, ['energy', 'forces'])
    en2 = atoms.get_potential_energy()
    assert en1 - en2 > 1e-07
    calc.kpts = 2
    assert calc.check_state(atoms) == ['input_params']
    assert calc.calculation_required(atoms, ['energy', 'forces'])
    calc.clean()