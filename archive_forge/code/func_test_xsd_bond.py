def test_xsd_bond():
    from ase import Atoms
    from ase.io import write
    import numpy as np
    from collections import OrderedDict
    import re
    atoms = Atoms('CH4', [[1.08288111e-09, 1.74602682e-09, -1.54703448e-09], [-0.678446715, 0.873516584, -0.0863073811], [-0.409602527, -0.84601653, -0.589280858], [0.085201607, -0.298243876, 1.06515792], [1.00284763, 0.270743821, -0.389569679]])
    connectivitymatrix = np.array([[0, 1, 1, 1, 1], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0], [1, 0, 0, 0, 0]])
    write('xsd_test_CH4.xsd', atoms, connectivity=connectivitymatrix)
    AtomIdsToBondIds = OrderedDict()
    BondIdsToConnectedAtomIds = OrderedDict()
    with open('xsd_test_CH4.xsd', 'r') as fd:
        for i, line in enumerate(fd):
            if '<Atom3d ' in line:
                AtomId = int(re.search('ID="(.*?)"', line).group(1))
                ConnectedBondIds = [int(a) for a in re.search('Connections="(.*?)"', line).group(1).split(',')]
                AtomIdsToBondIds[AtomId] = ConnectedBondIds
            elif '<Bond ' in line:
                BondId = int(re.search('ID="(.*?)"', line).group(1))
                ConnectedAtomIds = [int(a) for a in re.search('Connects="(.*?)"', line).group(1).split(',')]
                BondIdsToConnectedAtomIds[BondId] = ConnectedAtomIds
    for AtomId in AtomIdsToBondIds:
        for BondId in AtomIdsToBondIds[AtomId]:
            assert AtomId in BondIdsToConnectedAtomIds[BondId]
    AtomIds = list(AtomIdsToBondIds.keys())
    Newconnectivitymatrix = np.zeros((5, 5))
    for AtomId in AtomIdsToBondIds:
        for BondId in AtomIdsToBondIds[AtomId]:
            OtherAtomId = [a for a in BondIdsToConnectedAtomIds[BondId] if a != AtomId]
            i = AtomIds.index(AtomId)
            j = AtomIds.index(OtherAtomId[0])
            Newconnectivitymatrix[i, j] = 1
    for i in range(0, 4):
        for j in range(0, 4):
            assert connectivitymatrix[i, j] == Newconnectivitymatrix[i, j]