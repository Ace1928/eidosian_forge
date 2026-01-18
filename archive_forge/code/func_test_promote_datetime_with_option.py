import pytest
from datashader.datashape import (
@pytest.mark.parametrize('x,y,p,r', [[datetime, datetime, True, datetime], [datetime, datetime, False, datetime], [Option(datetime), Option(datetime), True, Option(datetime)], [Option(datetime), Option(datetime), False, Option(datetime)], [Option(datetime), datetime, True, Option(datetime)], [Option(datetime), datetime, False, datetime], [Option(datetime), dshape('?datetime'), True, Option(datetime)], [dshape('?datetime'), Option(datetime), False, Option(datetime)], [dshape('datetime'), Option(datetime), True, Option(datetime)], [dshape('datetime'), Option(datetime), False, datetime]])
def test_promote_datetime_with_option(x, y, p, r):
    assert promote(x, y, promote_option=p) == promote(y, x, promote_option=p) == r