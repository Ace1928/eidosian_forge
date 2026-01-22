from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
class TestRDSLogFileDownload(AWSMockServiceTestCase):
    connection_class = RDSConnection
    logfile_sample = "\n??2014-01-26 23:59:00.01 spid54      Microsoft SQL Server 2012 - 11.0.2100.60 (X64) \n    \n    Feb 10 2012 19:39:15 \n    \n    Copyright (c) Microsoft Corporation\n    \n    Web Edition (64-bit) on Windows NT 6.1 &lt;X64&gt; (Build 7601: Service Pack 1) (Hypervisor)\n    \n    \n    \n2014-01-26 23:59:00.01 spid54      (c) Microsoft Corporation.\n\n2014-01-26 23:59:00.01 spid54      All rights reserved.\n\n2014-01-26 23:59:00.01 spid54      Server process ID is 2976.\n\n2014-01-26 23:59:00.01 spid54      System Manufacturer: 'Xen', System Model: 'HVM domU'.\n\n2014-01-26 23:59:00.01 spid54      Authentication mode is MIXED.\n\n2014-01-26 23:59:00.01 spid54      Logging SQL Server messages in file 'D:\\RDSDBDATA\\Log\\ERROR'.\n\n2014-01-26 23:59:00.01 spid54      The service account is 'WORKGROUP\\AMAZONA-NUQUUMV$'. This is an informational message; no user action is required.\n\n2014-01-26 23:59:00.01 spid54      The error log has been reinitialized. See the previous log for older entries.\n\n2014-01-27 00:00:56.42 spid25s     This instance of SQL Server has been using a process ID of 2976 since 10/21/2013 2:16:50 AM (local) 10/21/2013 2:16:50 AM (UTC). This is an informational message only; no user action is required.\n\n2014-01-27 09:35:15.43 spid71      I/O is frozen on database model. No user action is required. However, if I/O is not resumed promptly, you could cancel the backup.\n\n2014-01-27 09:35:15.44 spid72      I/O is frozen on database msdb. No user action is required. However, if I/O is not resumed promptly, you could cancel the backup.\n\n2014-01-27 09:35:15.44 spid74      I/O is frozen on database rdsadmin. No user action is required. However, if I/O is not resumed promptly, you could cancel the backup.\n\n2014-01-27 09:35:15.44 spid73      I/O is frozen on database master. No user action is required. However, if I/O is not resumed promptly, you could cancel the backup.\n\n2014-01-27 09:35:25.57 spid73      I/O was resumed on database master. No user action is required.\n\n2014-01-27 09:35:25.57 spid74      I/O was resumed on database rdsadmin. No user action is required.\n\n2014-01-27 09:35:25.57 spid71      I/O was resumed on database model. No user action is required.\n\n2014-01-27 09:35:25.57 spid72      I/O was resumed on database msdb. No user action is required.\n    "

    def setUp(self):
        super(TestRDSLogFileDownload, self).setUp()

    def default_body(self):
        return '\n<DownloadDBLogFilePortionResponse xmlns="http://rds.amazonaws.com/doc/2013-09-09/">\n  <DownloadDBLogFilePortionResult>\n    <Marker>0:4485</Marker>\n    <LogFileData>%s</LogFileData>\n    <AdditionalDataPending>false</AdditionalDataPending>\n  </DownloadDBLogFilePortionResult>\n  <ResponseMetadata>\n    <RequestId>27143615-87ae-11e3-acc9-fb64b157268e</RequestId>\n  </ResponseMetadata>\n</DownloadDBLogFilePortionResponse>\n        ' % self.logfile_sample

    def test_single_download(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_log_file('db1', 'foo.log')
        self.assertTrue(isinstance(response, LogFileObject))
        self.assertEqual(response.marker, '0:4485')
        self.assertEqual(response.dbinstance_id, 'db1')
        self.assertEqual(response.log_filename, 'foo.log')
        self.assertEqual(response.data, saxutils.unescape(self.logfile_sample))
        self.assert_request_parameters({'Action': 'DownloadDBLogFilePortion', 'DBInstanceIdentifier': 'db1', 'LogFileName': 'foo.log'}, ignore_params_values=['Version'])

    def test_multi_args(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.get_log_file('db1', 'foo.log', marker='0:4485', number_of_lines=10)
        self.assertTrue(isinstance(response, LogFileObject))
        self.assert_request_parameters({'Action': 'DownloadDBLogFilePortion', 'DBInstanceIdentifier': 'db1', 'Marker': '0:4485', 'NumberOfLines': 10, 'LogFileName': 'foo.log'}, ignore_params_values=['Version'])