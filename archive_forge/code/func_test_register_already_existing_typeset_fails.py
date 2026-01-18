from datashader import datashape
import pytest
def test_register_already_existing_typeset_fails():
    mytypeset = datashape.TypeSet(datashape.int64, datashape.float64, name='foo')
    with pytest.raises(TypeError):
        datashape.typesets.register_typeset('foo', mytypeset)