from __future__ import absolute_import, print_function, division
import pytest
from petl.test.helpers import eq_, ieq, get_env_vars_named
def _testcase_get_env_vars_named(num_vals, prefix=''):
    res = {}
    for i in range(1, num_vals, 1):
        reskey = prefix + str(i)
        res[reskey] = str(i)
    return res