from tests.unit import unittest
from tests.unit import AWSMockServiceTestCase
from boto.rds import RDSConnection
from boto.rds.dbsnapshot import DBSnapshot
from boto.rds import DBInstance
class TestDeleteDBSnapshot(AWSMockServiceTestCase):
    connection_class = RDSConnection

    def default_body(self):
        return '\n        <DeleteDBSnapshotResponse xmlns="http://rds.amazonaws.com/doc/2013-05-15/">\n            <DeleteDBSnapshotResult>\n                <DBSnapshot>\n                <Port>3306</Port>\n                <SnapshotCreateTime>2011-03-11T07:20:24.082Z</SnapshotCreateTime>\n                <Engine>mysql</Engine>\n                <Status>deleted</Status>\n                <AvailabilityZone>us-east-1d</AvailabilityZone>\n                <LicenseModel>general-public-license</LicenseModel>\n                <InstanceCreateTime>2010-07-16T00:06:59.107Z</InstanceCreateTime>\n                <AllocatedStorage>60</AllocatedStorage>\n                <DBInstanceIdentifier>simcoprod01</DBInstanceIdentifier>\n                <EngineVersion>5.1.47</EngineVersion>\n                <DBSnapshotIdentifier>mysnapshot2</DBSnapshotIdentifier>\n                <SnapshotType>manual</SnapshotType>\n                <MasterUsername>master</MasterUsername>\n                </DBSnapshot>\n            </DeleteDBSnapshotResult>\n            <ResponseMetadata>\n                <RequestId>627a43a1-8507-11e0-bd9b-a7b1ece36d51</RequestId>\n            </ResponseMetadata>\n        </DeleteDBSnapshotResponse>\n        '

    def test_delete_dbinstance(self):
        self.set_http_response(status_code=200)
        response = self.service_connection.delete_dbsnapshot('mysnapshot2')
        self.assert_request_parameters({'Action': 'DeleteDBSnapshot', 'DBSnapshotIdentifier': 'mysnapshot2'}, ignore_params_values=['Version'])
        self.assertIsInstance(response, DBSnapshot)
        self.assertEqual(response.id, 'mysnapshot2')
        self.assertEqual(response.status, 'deleted')