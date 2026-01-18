import pytest
from datashader import datashape
from datashader.datashape import dshape, has_var_dim, has_ellipsis
@pytest.mark.parametrize('ds_neg', [dshape('float32'), dshape('10 * float32'), dshape('10 * { f0: int32, f1: 10 * float32 }'), dshape('{ f0 : { g0 : 2 * int }, f1: int32 }'), (dshape('int32'),)])
def test_not_has_var_dim(ds_neg):
    assert not has_var_dim(ds_neg)