import warnings
import pytest
import numpy as np
import matplotlib as mpl
from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import matplotlib.category as cat
from matplotlib.testing.decorators import check_figures_equal
class TestUnitData:
    test_cases = [('single', (['hello world'], [0])), ('unicode', (['Здравствуйте мир'], [0])), ('mixed', (['A', 'np.nan', 'B', '3.14', 'мир'], [0, 1, 2, 3, 4]))]
    ids, data = zip(*test_cases)

    @pytest.mark.parametrize('data, locs', data, ids=ids)
    def test_unit(self, data, locs):
        unit = cat.UnitData(data)
        assert list(unit._mapping.keys()) == data
        assert list(unit._mapping.values()) == locs

    def test_update(self):
        data = ['a', 'd']
        locs = [0, 1]
        data_update = ['b', 'd', 'e']
        unique_data = ['a', 'd', 'b', 'e']
        updated_locs = [0, 1, 2, 3]
        unit = cat.UnitData(data)
        assert list(unit._mapping.keys()) == data
        assert list(unit._mapping.values()) == locs
        unit.update(data_update)
        assert list(unit._mapping.keys()) == unique_data
        assert list(unit._mapping.values()) == updated_locs
    failing_test_cases = [('number', 3.14), ('nan', np.nan), ('list', [3.14, 12]), ('mixed type', ['A', 2])]
    fids, fdata = zip(*test_cases)

    @pytest.mark.parametrize('fdata', fdata, ids=fids)
    def test_non_string_fails(self, fdata):
        with pytest.raises(TypeError):
            cat.UnitData(fdata)

    @pytest.mark.parametrize('fdata', fdata, ids=fids)
    def test_non_string_update_fails(self, fdata):
        unitdata = cat.UnitData()
        with pytest.raises(TypeError):
            unitdata.update(fdata)