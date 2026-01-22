from datetime import datetime
from urllib import parse as urlparse
from cinderclient import client as base_client
from cinderclient.tests.unit import fakes
import cinderclient.tests.unit.utils as utils
def _cs_request(self, url, method, **kwargs):
    if method in ['GET', 'DELETE']:
        assert 'body' not in kwargs
    elif method == 'PUT':
        assert 'body' in kwargs
    args = urlparse.parse_qsl(urlparse.urlparse(url)[4])
    kwargs.update(args)
    url_split = url.rsplit('?', 1)
    munged_url = url_split[0]
    if len(url_split) > 1:
        parameters = url_split[1]
        if 'marker' in parameters:
            self.marker = int(parameters.rsplit('marker=', 1)[1])
        else:
            self.marker = None
    else:
        self.marker = None
    munged_url = munged_url.strip('/').replace('/', '_').replace('.', '_')
    munged_url = munged_url.replace('-', '_')
    callback = '%s_%s' % (method.lower(), munged_url)
    if not hasattr(self, callback):
        raise AssertionError('Called unknown API method: %s %s, expected fakes method name: %s' % (method, url, callback))
    self.callstack.append((method, url, kwargs.get('body', None)))
    status, headers, body = getattr(self, callback)(**kwargs)
    headers['x-openstack-request-id'] = REQUEST_ID
    if self.version_header:
        headers['OpenStack-API-version'] = self.version_header
    r = utils.TestResponse({'status_code': status, 'text': body, 'headers': headers})
    return (r, body)