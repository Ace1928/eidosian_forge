from unittest import mock
import uuid
from oslotest import base as test_base
import statsd
import webob.dec
import webob.exc
from oslo_middleware import stats
def test_removes_uuid(self):
    id = str(uuid.uuid4())
    path = 'foo.{uuid}.bar'.format(uuid=id)
    stat = stats.StatsMiddleware.strip_uuid(path)
    self.assertEqual('foo.bar', stat)