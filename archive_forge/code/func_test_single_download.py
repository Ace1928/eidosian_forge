from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
def test_single_download(self):
    self.set_http_response(status_code=200)
    response = self.service_connection.get_log_file('db1', 'foo.log')
    self.assertTrue(isinstance(response, LogFileObject))
    self.assertEqual(response.marker, '0:4485')
    self.assertEqual(response.dbinstance_id, 'db1')
    self.assertEqual(response.log_filename, 'foo.log')
    self.assertEqual(response.data, saxutils.unescape(self.logfile_sample))
    self.assert_request_parameters({'Action': 'DownloadDBLogFilePortion', 'DBInstanceIdentifier': 'db1', 'LogFileName': 'foo.log'}, ignore_params_values=['Version'])