from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.ec2.securitygroup import SecurityGroup
from boto.rds import RDSConnection
from boto.rds.vpcsecuritygroupmembership import VPCSecurityGroupMembership
from boto.rds.parametergroup import ParameterGroup
from boto.rds.logfile import LogFile, LogFileObject
import xml.sax.saxutils as saxutils
class TestRDSConnectionRestoreDBInstanceFromPointInTime(AWSMockServiceTestCase):
    connection_class = RDSConnection

    def setUp(self):
        super(TestRDSConnectionRestoreDBInstanceFromPointInTime, self).setUp()

    def default_body(self):
        return '\n        <RestoreDBInstanceToPointInTimeResponse xmlns="http://rds.amazonaws.com/doc/2013-05-15/">\n          <RestoreDBInstanceToPointInTimeResult>\n            <DBInstance>\n              <ReadReplicaDBInstanceIdentifiers/>\n              <Engine>mysql</Engine>\n              <PendingModifiedValues/>\n              <BackupRetentionPeriod>1</BackupRetentionPeriod>\n              <MultiAZ>false</MultiAZ>\n              <LicenseModel>general-public-license</LicenseModel>\n              <DBInstanceStatus>creating</DBInstanceStatus>\n              <EngineVersion>5.1.50</EngineVersion>\n              <DBInstanceIdentifier>restored-db</DBInstanceIdentifier>\n              <DBParameterGroups>\n                <DBParameterGroup>\n                  <ParameterApplyStatus>in-sync</ParameterApplyStatus>\n                  <DBParameterGroupName>default.mysql5.1</DBParameterGroupName>\n                </DBParameterGroup>\n              </DBParameterGroups>\n              <DBSecurityGroups>\n                <DBSecurityGroup>\n                  <Status>active</Status>\n                  <DBSecurityGroupName>default</DBSecurityGroupName>\n                </DBSecurityGroup>\n              </DBSecurityGroups>\n              <PreferredBackupWindow>00:00-00:30</PreferredBackupWindow>\n              <AutoMinorVersionUpgrade>true</AutoMinorVersionUpgrade>\n              <PreferredMaintenanceWindow>sat:07:30-sat:08:00</PreferredMaintenanceWindow>\n              <AllocatedStorage>10</AllocatedStorage>\n              <DBInstanceClass>db.m1.large</DBInstanceClass>\n              <MasterUsername>master</MasterUsername>\n            </DBInstance>\n          </RestoreDBInstanceToPointInTimeResult>\n          <ResponseMetadata>\n            <RequestId>1ef546bc-850b-11e0-90aa-eb648410240d</RequestId>\n          </ResponseMetadata>\n        </RestoreDBInstanceToPointInTimeResponse>\n        '

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

    def test_restore_dbinstance_from_point_in_time__db_subnet_group_name(self):
        self.set_http_response(status_code=200)
        db = self.service_connection.restore_dbinstance_from_point_in_time('simcoprod01', 'restored-db', True, db_subnet_group_name='dbsubnetgroup')
        self.assert_request_parameters({'Action': 'RestoreDBInstanceToPointInTime', 'SourceDBInstanceIdentifier': 'simcoprod01', 'TargetDBInstanceIdentifier': 'restored-db', 'UseLatestRestorableTime': 'true', 'DBSubnetGroupName': 'dbsubnetgroup'}, ignore_params_values=['Version'])

    def test_create_db_instance_vpc_sg_str(self):
        self.set_http_response(status_code=200)
        vpc_security_groups = [VPCSecurityGroupMembership(self.service_connection, 'active', 'sg-1'), VPCSecurityGroupMembership(self.service_connection, None, 'sg-2')]
        db = self.service_connection.create_dbinstance('SimCoProd01', 10, 'db.m1.large', 'master', 'Password01', param_group='default.mysql5.1', db_subnet_group_name='dbSubnetgroup01', vpc_security_groups=vpc_security_groups)
        self.assert_request_parameters({'Action': 'CreateDBInstance', 'AllocatedStorage': 10, 'AutoMinorVersionUpgrade': 'true', 'DBInstanceClass': 'db.m1.large', 'DBInstanceIdentifier': 'SimCoProd01', 'DBParameterGroupName': 'default.mysql5.1', 'DBSubnetGroupName': 'dbSubnetgroup01', 'Engine': 'MySQL5.1', 'MasterUsername': 'master', 'MasterUserPassword': 'Password01', 'Port': 3306, 'VpcSecurityGroupIds.member.1': 'sg-1', 'VpcSecurityGroupIds.member.2': 'sg-2'}, ignore_params_values=['Version'])

    def test_create_db_instance_vpc_sg_obj(self):
        self.set_http_response(status_code=200)
        sg1 = SecurityGroup(name='sg-1')
        sg2 = SecurityGroup(name='sg-2')
        vpc_security_groups = [VPCSecurityGroupMembership(self.service_connection, 'active', sg1.name), VPCSecurityGroupMembership(self.service_connection, None, sg2.name)]
        db = self.service_connection.create_dbinstance('SimCoProd01', 10, 'db.m1.large', 'master', 'Password01', param_group='default.mysql5.1', db_subnet_group_name='dbSubnetgroup01', vpc_security_groups=vpc_security_groups)
        self.assert_request_parameters({'Action': 'CreateDBInstance', 'AllocatedStorage': 10, 'AutoMinorVersionUpgrade': 'true', 'DBInstanceClass': 'db.m1.large', 'DBInstanceIdentifier': 'SimCoProd01', 'DBParameterGroupName': 'default.mysql5.1', 'DBSubnetGroupName': 'dbSubnetgroup01', 'Engine': 'MySQL5.1', 'MasterUsername': 'master', 'MasterUserPassword': 'Password01', 'Port': 3306, 'VpcSecurityGroupIds.member.1': 'sg-1', 'VpcSecurityGroupIds.member.2': 'sg-2'}, ignore_params_values=['Version'])