from tests.compat import unittest
from boto.s3.connection import S3Connection
from boto.s3 import connect_to_region
class S3SpecifyHost(unittest.TestCase):
    s3 = True

    def testWithNonAWSHost(self):
        connect_args = dict({'host': 'www.not-a-website.com'})
        connection = connect_to_region('us-east-1', **connect_args)
        self.assertEquals('www.not-a-website.com', connection.host)
        self.assertIsInstance(connection, S3Connection)

    def testSuccessWithHostOverrideRegion(self):
        connect_args = dict({'host': 's3.amazonaws.com'})
        connection = connect_to_region('us-west-2', **connect_args)
        self.assertEquals('s3.amazonaws.com', connection.host)
        self.assertIsInstance(connection, S3Connection)

    def testSuccessWithDefaultUSWest1(self):
        connection = connect_to_region('us-west-2')
        self.assertEquals('s3-us-west-2.amazonaws.com', connection.host)
        self.assertIsInstance(connection, S3Connection)

    def testSuccessWithDefaultUSEast1(self):
        connection = connect_to_region('us-east-1')
        self.assertEquals('s3.amazonaws.com', connection.host)
        self.assertIsInstance(connection, S3Connection)

    def testSuccessWithDefaultEUCentral1(self):
        connection = connect_to_region('eu-central-1')
        self.assertEquals('s3.eu-central-1.amazonaws.com', connection.host)
        self.assertIsInstance(connection, S3Connection)

    def testDefaultWithInvalidHost(self):
        connect_args = dict({'host': ''})
        connection = connect_to_region('us-west-2', **connect_args)
        self.assertEquals('s3-us-west-2.amazonaws.com', connection.host)
        self.assertIsInstance(connection, S3Connection)

    def testDefaultWithInvalidHostNone(self):
        connect_args = dict({'host': None})
        connection = connect_to_region('us-east-1', **connect_args)
        self.assertEquals('s3.amazonaws.com', connection.host)
        self.assertIsInstance(connection, S3Connection)

    def tearDown(self):
        self = connection = connect_args = None