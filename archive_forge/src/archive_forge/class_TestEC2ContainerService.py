import boto
from boto.ec2containerservice.exceptions import ClientException
from tests.compat import unittest
class TestEC2ContainerService(unittest.TestCase):

    def setUp(self):
        self.ecs = boto.connect_ec2containerservice()

    def test_list_clusters(self):
        response = self.ecs.list_clusters()
        self.assertIn('clusterArns', response['ListClustersResponse']['ListClustersResult'])

    def test_handle_not_found_exception(self):
        with self.assertRaises(ClientException):
            self.ecs.stop_task(task='foo')