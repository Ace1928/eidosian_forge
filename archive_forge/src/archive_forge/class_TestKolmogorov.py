import itertools
import sys
import pytest
import numpy as np
from numpy.testing import assert_
from scipy.special._testutils import FuncData
from scipy.special import kolmogorov, kolmogi, smirnov, smirnovi
from scipy.special._ufuncs import (_kolmogc, _kolmogci, _kolmogp,
class TestKolmogorov:

    def test_nan(self):
        assert_(np.isnan(kolmogorov(np.nan)))

    def test_basic(self):
        dataset = [(0, 1.0), (0.5, 0.9639452436648751), (0.8275735551899077, 0.5), (1, 0.26999967167735456), (2, 0.0006709252557796953)]
        dataset = np.asarray(dataset)
        FuncData(kolmogorov, dataset, (0,), 1, rtol=_rtol).check()

    def test_linspace(self):
        x = np.linspace(0, 2.0, 21)
        dataset = [1.0, 1.0, 0.999999999999495, 0.9999906941986655, 0.9971923267772983, 0.9639452436648751, 0.8642827790506042, 0.711235195029689, 0.5441424115741981, 0.3927307079406543, 0.2699996716773546, 0.1777181926064012, 0.1122496666707249, 0.0680922218447664, 0.0396818795381144, 0.0222179626165251, 0.0119520432391966, 0.0061774306344441, 0.0030676213475797, 0.0014636048371873, 0.0006709252557797]
        dataset_c = [0.0, 6.609305242245699e-53, 5.050407338670114e-13, 9.305801334566668e-06, 0.0028076732227017, 0.0360547563351249, 0.1357172209493958, 0.288764804970311, 0.4558575884258019, 0.6072692920593457, 0.7300003283226455, 0.8222818073935988, 0.8877503333292751, 0.9319077781552336, 0.9603181204618857, 0.9777820373834749, 0.9880479567608034, 0.9938225693655559, 0.9969323786524203, 0.9985363951628127, 0.9993290747442203]
        dataset = np.column_stack([x, dataset])
        FuncData(kolmogorov, dataset, (0,), 1, rtol=_rtol).check()
        dataset_c = np.column_stack([x, dataset_c])
        FuncData(_kolmogc, dataset_c, (0,), 1, rtol=_rtol).check()

    def test_linspacei(self):
        p = np.linspace(0, 1.0, 21, endpoint=True)
        dataset = [np.inf, 1.3580986393225507, 1.2238478702170823, 1.1379465424937751, 1.072749174939648, 1.019184720253686, 0.9730633753323726, 0.9320695842357622, 0.8947644549851197, 0.8601710725555463, 0.8275735551899077, 0.7964065373291559, 0.7661855555617682, 0.736454288817191, 0.706732652306898, 0.6764476915028201, 0.6448126061663567, 0.6105590999244391, 0.5711732651063401, 0.5196103791686224, 0.0]
        dataset_c = [0.0, 0.5196103791686225, 0.5711732651063401, 0.6105590999244391, 0.6448126061663567, 0.6764476915028201, 0.706732652306898, 0.736454288817191, 0.7661855555617682, 0.7964065373291559, 0.8275735551899077, 0.8601710725555463, 0.8947644549851196, 0.9320695842357622, 0.9730633753323727, 1.019184720253686, 1.072749174939648, 1.1379465424937754, 1.2238478702170825, 1.358098639322551, np.inf]
        dataset = np.column_stack([p[1:], dataset[1:]])
        FuncData(kolmogi, dataset, (0,), 1, rtol=_rtol).check()
        dataset_c = np.column_stack([p[:-1], dataset_c[:-1]])
        FuncData(_kolmogci, dataset_c, (0,), 1, rtol=_rtol).check()

    def test_smallx(self):
        epsilon = 0.1 ** np.arange(1, 14)
        x = np.array([0.571173265106, 0.441027698518, 0.374219690278, 0.331392659217, 0.300820537459, 0.277539353999, 0.259023494805, 0.243829561254, 0.231063086389, 0.220135543236, 0.210641372041, 0.202290283658, 0.19487060742])
        dataset = np.column_stack([x, 1 - epsilon])
        FuncData(kolmogorov, dataset, (0,), 1, rtol=_rtol).check()

    def test_round_trip(self):

        def _ki_k(_x):
            return kolmogi(kolmogorov(_x))

        def _kci_kc(_x):
            return _kolmogci(_kolmogc(_x))
        x = np.linspace(0.0, 2.0, 21, endpoint=True)
        x02 = x[(x == 0) | (x > 0.21)]
        dataset02 = np.column_stack([x02, x02])
        FuncData(_ki_k, dataset02, (0,), 1, rtol=_rtol).check()
        dataset = np.column_stack([x, x])
        FuncData(_kci_kc, dataset, (0,), 1, rtol=_rtol).check()