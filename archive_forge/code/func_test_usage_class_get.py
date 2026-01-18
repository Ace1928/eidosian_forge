import datetime
from novaclient import api_versions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import usage
def test_usage_class_get(self):
    start = '2012-01-22T19:48:41.750722'
    stop = '2012-01-22T19:48:41.750722'
    info = {'tenant_id': 'tenantfoo', 'start': start, 'stop': stop}
    u = usage.Usage(self.cs.usage, info)
    u.get()
    self.assert_request_id(u, fakes.FAKE_REQUEST_ID_LIST)
    self.cs.assert_called('GET', '/os-simple-tenant-usage/tenantfoo?start=%s&end=%s' % (start, stop))