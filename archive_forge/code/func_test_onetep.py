def test_onetep(testdir):
    from ase.build import molecule
    from ase.calculators.onetep import Onetep
    from os.path import isfile, join
    mol = molecule('H2O')
    mol.center(8)
    calc = Onetep(label='water')
    prefix = 'hello'
    h_path = join(prefix, 'H.abinit')
    o_path = join(prefix, 'O.abinit')
    if not (isfile(h_path) and isfile(o_path)):
        raise Exception('You must supply PAW data sets for\n            hydrogen and oxygen to run this test.\n            Please see http://www.abinit.org/downloads/PAW2\n            for suitable data. ONETEP takes PAW data sets in the\n            abinit format. I need H.abinit and O.abinit')
    calc.set_pseudos([('H', h_path), ('O', o_path)])
    calc.set(paw=True, xc='PBE', cutoff_energy='400 eV')
    mol.calc = calc
    energy = mol.get_total_energy()
    ref_energy = -470.852068717
    assert abs(energy - ref_energy) < 1e-06