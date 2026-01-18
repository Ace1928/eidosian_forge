import testtools
from neutronclient.common import exceptions
from neutronclient.common import validators
def test_validate_int_min_max(self):
    self._test_validate_int(1)
    self._test_validate_int(10)
    self._test_validate_int('1')
    self._test_validate_int('10')
    self._test_validate_int('0x0a')
    self._test_validate_int_error(0, 'attr1 "0" should be an integer [1:10].')
    self._test_validate_int_error(11, 'attr1 "11" should be an integer [1:10].')
    self._test_validate_int_error('0x10', 'attr1 "0x10" should be an integer [1:10].')