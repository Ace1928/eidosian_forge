import pytest
from datashader.datashape import (
@pytest.mark.parametrize('x,y,p,r', [[string, string, True, string], [string, string, False, string], [Option(string), Option(string), True, Option(string)], [Option(string), Option(string), False, Option(string)], [Option(string), string, True, Option(string)], [Option(string), string, False, string], [Option(string), dshape('?string'), True, Option(string)], [dshape('?string'), Option(string), False, Option(string)], [dshape('string'), Option(string), True, Option(string)], [dshape('string'), Option(string), False, string]])
def test_promote_string_with_option(x, y, p, r):
    assert promote(x, y, promote_option=p) == promote(y, x, promote_option=p) == r