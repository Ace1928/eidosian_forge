import os
import pyomo.common.unittest as unittest
def test_set_odbc_info(self):
    config = ODBCConfig()
    config.set_odbc_info('UNICODE', 'UTF-8')
    self.assertEqual({'UNICODE': 'UTF-8'}, config.odbc_info)