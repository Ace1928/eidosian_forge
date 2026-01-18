import os
import pytest
@pytest.mark.filterwarnings('ignore:Specifying directory')
@calc('vasp')
def test_vasp_wdir(factory, atoms_co):
    """
    Run tests to ensure that the VASP txt and label arguments function correctly,
    i.e. correctly sets the working directories and works in that directory.

    This is conditional on the existence of the ASE_VASP_COMMAND, VASP_COMMAND
    or VASP_SCRIPT environment variables

    """

    def compare_paths(path1, path2):
        assert os.path.abspath(path1) == os.path.abspath(path2)
    atoms = atoms_co
    file1 = '_vasp_dummy_str.out'
    file2 = '_vasp_dummy_io.out'
    file3 = '_vasp_dummy_2.out'
    testdir = '_dummy_txt_testdir'
    label = os.path.join(testdir, 'vasp')
    settings = dict(label=label, xc='PBE', prec='Low', algo='Fast', ismear=0, sigma=1.0, istart=0, lwave=False, lcharg=False)
    calc = factory.calc(**settings)
    calc2 = factory.calc(**settings)
    compare_paths(calc.directory, testdir)
    calc.set(txt=file1)
    atoms.calc = calc
    en1 = atoms.get_potential_energy()
    for fi in ['OUTCAR', 'CONTCAR', 'vasprun.xml']:
        fi = os.path.join(testdir, fi)
        assert os.path.isfile(fi)
    with open(file2, 'w') as fd:
        calc2.set(txt=fd)
        atoms.calc = calc2
        atoms.get_potential_energy()
    label2 = os.path.join(testdir, file3)
    calc2 = factory.calc(restart=label, label=label2)
    compare_paths(calc2.directory, testdir)
    assert not calc2.calculation_required(calc2.atoms, ['energy', 'forces'])
    en2 = calc2.get_potential_energy()
    assert not os.path.isfile(os.path.join(calc.directory, file3))
    assert en1 == en2