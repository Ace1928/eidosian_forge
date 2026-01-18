import re
import numpy as np
from numpy.testing import assert_
def test_valid_numpy_version():
    version_pattern = '^[0-9]+\\.[0-9]+\\.[0-9]+(a[0-9]|b[0-9]|rc[0-9])?'
    dev_suffix = '(\\.dev[0-9]+(\\+git[0-9]+\\.[0-9a-f]+)?)?'
    res = re.match(version_pattern + dev_suffix + '$', np.__version__)
    assert_(res is not None, np.__version__)