from unittest import mock
import uuid
from oslotest import base as test_base
import statsd
import webob.dec
import webob.exc
from oslo_middleware import stats
def test_strips_dots_from_version(self):
    path = '/v1.2/foo.bar/bar.foo'
    stat = stats.StatsMiddleware.strip_dot_from_version(path)
    self.assertEqual('/v12/foo.bar/bar.foo', stat)