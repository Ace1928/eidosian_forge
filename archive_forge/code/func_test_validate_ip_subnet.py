import testtools
from neutronclient.common import exceptions
from neutronclient.common import validators
def test_validate_ip_subnet(self):
    self._test_validate_subnet('192.168.2.0/24')
    self._test_validate_subnet('192.168.2.3/20')
    self._test_validate_subnet('192.168.2.1')
    e = self.assertRaises(exceptions.CommandError, self._test_validate_subnet, '192.168.2.256')
    self.assertEqual('attr1 "192.168.2.256" is not a valid CIDR.', str(e))