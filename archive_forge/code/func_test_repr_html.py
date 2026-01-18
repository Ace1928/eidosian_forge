import pytest
import rpy2.robjects as robjects
import os
import array
import time
import datetime
import rpy2.rlike.container as rlc
from collections import OrderedDict
@pytest.mark.parametrize('params', ((robjects.vectors.DataFrame, dict((('a', ri.IntSexpVector((1, 2, 3))), ('b', ri.IntSexpVector((4, 5, 6))), ('c', ri.IntSexpVector((7, 8, 9))), ('d', ri.IntSexpVector((7, 8, 9))), ('e', ri.IntSexpVector((7, 8, 9)))))), (robjects.vectors.IntVector, (1, 2, 3, 4, 5)), (robjects.vectors.ListVector, (('a', 1), ('b', 2), ('b', 3), ('c', 4), ('d', 5))), (robjects.vectors.FloatVector, (1, 2, 3, 4, 5))))
def test_repr_html(params):
    vec_cls, data = params
    vec = vec_cls(data)
    s = vec._repr_html_().split('\n')
    assert s[2].strip().startswith('<table>')
    assert s[-2].strip().endswith('</table>')
    s = vec._repr_html_(max_items=2).split('\n')
    assert s[2].strip().startswith('<table>')
    assert s[-2].strip().endswith('</table>')