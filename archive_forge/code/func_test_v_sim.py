from ase.io import read
def test_v_sim():
    fname = 'demo.ascii'
    copy = 'demo2.ascii'
    with open(fname, 'w') as fd:
        fd.write(datafile)
    atoms = read(fname, format='v-sim')
    atoms.write(copy)
    atoms2 = read(copy)
    tol = 1e-06
    assert sum(abs((atoms.positions - atoms2.positions).ravel())) < tol
    assert sum(abs((atoms.cell - atoms2.cell).ravel())) < tol