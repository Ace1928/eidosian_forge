from unittest import mock
import uuid
from oslotest import base as test_base
import statsd
import webob.dec
import webob.exc
from oslo_middleware import stats
def test_sends_counter_to_statsd(self):
    app = self.make_stats_middleware()
    path = '/test/foo/bar'
    self.perform_request(app, path, 'GET')
    expected_stat = '{name}.{method}.{path}'.format(name=app.stat_name, method='GET', path=path.lstrip('/').replace('/', '.'))
    app.statsd.timer.assert_called_once_with(expected_stat)