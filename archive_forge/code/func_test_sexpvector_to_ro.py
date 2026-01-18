import array
import pytest
import rpy2.rinterface_lib.sexp
from rpy2 import rinterface
from rpy2 import robjects
from rpy2.robjects import conversion
@pytest.mark.parametrize('cls, values, expected_cls', [(rinterface.IntSexpVector, (1, 2), robjects.vectors.IntVector), (rinterface.FloatSexpVector, (1.1, 2.2), robjects.vectors.FloatVector), (rinterface.StrSexpVector, ('ab', 'cd'), robjects.vectors.StrVector), (rinterface.BoolSexpVector, (True, False), robjects.vectors.BoolVector), (rinterface.ByteSexpVector, b'ab', robjects.vectors.ByteVector), (lambda x: rinterface.evalr(x), 'y ~ x', robjects.Formula)])
def test_sexpvector_to_ro(cls, values, expected_cls):
    v_ri = cls(values)
    v_ro = robjects.default_converter.rpy2py(v_ri)
    assert isinstance(v_ro, expected_cls)