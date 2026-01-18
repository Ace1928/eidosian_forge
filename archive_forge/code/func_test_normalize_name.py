import re
import os_traits as ot
from os_traits.hw.cpu import x86
from os_traits.hw.gpu import api
from os_traits.hw.gpu import resolution
from os_traits.hw.nic import offload
from os_traits.tests import base
def test_normalize_name(self):
    values = [('foo', 'CUSTOM_FOO'), ('VCPU', 'CUSTOM_VCPU'), ('CUSTOM_BOB', 'CUSTOM_CUSTOM_BOB'), ('CUSTM_BOB', 'CUSTOM_CUSTM_BOB'), (u'Fußball', u'CUSTOM_FU_BALL'), ('abc-123', 'CUSTOM_ABC_123'), ('Hello, world!  This is a test ^_^', 'CUSTOM_HELLO_WORLD_THIS_IS_A_TEST_'), ('  leading and trailing spaces  ', 'CUSTOM__LEADING_AND_TRAILING_SPACES_')]
    for test_value, expected in values:
        result = ot.normalize_name(test_value)
        self.assertEqual(expected, result)