import pytest
from datashader import datashape
from datashader.datashape import dshape, has_var_dim, has_ellipsis
@pytest.mark.parametrize('ds', [dshape('float32'), dshape('10 * var * float32'), dshape('M * float32'), dshape('(int32, M * int16) -> var * int8'), dshape('(int32, int16) -> var * int8'), dshape('10 * { f0: int32, f1: 10 * float32 }'), dshape('{ f0 : { g0 : 2 * int }, f1: int32 }'), (dshape('M * int32'),)])
def test_not_has_ellipsis(ds):
    assert not has_ellipsis(ds)