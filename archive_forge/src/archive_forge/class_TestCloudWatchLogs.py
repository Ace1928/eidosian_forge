import boto
from tests.compat import unittest
class TestCloudWatchLogs(unittest.TestCase):

    def setUp(self):
        self.logs = boto.connect_logs()

    def test_logs(self):
        logs = self.logs
        response = logs.describe_log_groups(log_group_name_prefix='test')
        self.assertIsInstance(response['logGroups'], list)
        mfilter = '[ip, id, user, ..., status_code=500, size]'
        sample = ['127.0.0.1 - frank "GET /apache_pb.gif HTTP/1.0" 200 1534', '127.0.0.1 - frank "GET /apache_pb.gif HTTP/1.0" 500 5324']
        response = logs.test_metric_filter(mfilter, sample)
        self.assertEqual(len(response['matches']), 1)