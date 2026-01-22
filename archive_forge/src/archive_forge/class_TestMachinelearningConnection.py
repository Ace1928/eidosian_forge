import os
from tests.unit import unittest
class TestMachinelearningConnection(unittest.TestCase):

    def test_connect_to_region(self):
        from boto.machinelearning import connect_to_region
        from boto.machinelearning.layer1 import MachineLearningConnection
        connection = connect_to_region('us-east-1', aws_access_key_id='foo', aws_secret_access_key='bar')
        self.assertIsInstance(connection, MachineLearningConnection)
        self.assertEqual(connection.host, 'machinelearning.us-east-1.amazonaws.com')