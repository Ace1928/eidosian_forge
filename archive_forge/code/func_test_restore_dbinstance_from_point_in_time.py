from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
def test_restore_dbinstance_from_point_in_time(self):
    self.set_http_response(status_code=200)
    db = self.service_connection.restore_dbinstance_from_point_in_time('simcoprod01', 'restored-db', True)
    self.assert_request_parameters({'Action': 'RestoreDBInstanceToPointInTime', 'SourceDBInstanceIdentifier': 'simcoprod01', 'TargetDBInstanceIdentifier': 'restored-db', 'UseLatestRestorableTime': 'true'}, ignore_params_values=['Version'])
    self.assertEqual(db.id, 'restored-db')
    self.assertEqual(db.engine, 'mysql')
    self.assertEqual(db.status, 'creating')
    self.assertEqual(db.allocated_storage, 10)
    self.assertEqual(db.instance_class, 'db.m1.large')
    self.assertEqual(db.master_username, 'master')
    self.assertEqual(db.multi_az, False)
    self.assertEqual(db.parameter_group.name, 'default.mysql5.1')
    self.assertEqual(db.parameter_group.description, None)
    self.assertEqual(db.parameter_group.engine, None)