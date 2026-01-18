from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
def test_restore_dbinstance_from_point_in_time__db_subnet_group_name(self):
    self.set_http_response(status_code=200)
    db = self.service_connection.restore_dbinstance_from_point_in_time('simcoprod01', 'restored-db', True, db_subnet_group_name='dbsubnetgroup')
    self.assert_request_parameters({'Action': 'RestoreDBInstanceToPointInTime', 'SourceDBInstanceIdentifier': 'simcoprod01', 'TargetDBInstanceIdentifier': 'restored-db', 'UseLatestRestorableTime': 'true', 'DBSubnetGroupName': 'dbsubnetgroup'}, ignore_params_values=['Version'])