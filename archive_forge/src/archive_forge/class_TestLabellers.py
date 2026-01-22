import pytest
from ...labels import (
@pytest.mark.parametrize('args', [('BaseLabeller', 'theta\na, 3'), ('DimCoordLabeller', 'theta\ninstrument: a, experiment: 3'), ('IdxLabeller', 'theta\n0, 4'), ('DimIdxLabeller', 'theta\ninstrument#0, experiment#4'), ('MapLabeller', 'theta\na, 3'), ('NoVarLabeller', 'a, 3'), ('NoModelLabeller', 'theta\na, 3')])
class TestLabellers:

    def test_make_label_vert(self, args, multidim_sels, labellers):
        name, expected_label = args
        labeller_arg = labellers.labellers[name]
        label = labeller_arg.make_label_vert('theta', multidim_sels.sel, multidim_sels.isel)
        assert label == expected_label