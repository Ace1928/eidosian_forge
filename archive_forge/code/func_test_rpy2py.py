import pytest
import sys
from rpy2 import robjects
from rpy2 import rinterface
import rpy2.rlike.container
import rpy2.robjects.conversion as conversion
@pytest.mark.skipif(not has_numpy, reason='package numpy cannot be imported')
@pytest.mark.parametrize('rcode,expected_values', (('c(TRUE, FALSE)', (True, False)), ('c(1, 2, 3)', (1, 2, 3)), ('c(1.0, 2.0, 3.0)', (1.0, 2.0, 3.0)), ('c("ab", "cd", NA_character_)', ('ab', 'cd', None))))
def test_rpy2py(rcode, expected_values):
    with (robjects.default_converter + rpyn.converter).context():
        values = robjects.r(rcode)
    assert tuple(values) == expected_values