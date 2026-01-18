import datetime
from boto.ec2.cloudwatch import CloudWatchConnection
from tests.compat import unittest, OrderedDict
def test_put_metric_data(self):
    c = CloudWatchConnection()
    now = datetime.datetime.utcnow()
    name, namespace = ('unit-test-metric', 'boto-unit-test')
    c.put_metric_data(namespace, name, 5, now, 'Bytes')