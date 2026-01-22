import functools
import re
import time
from urllib import parse
import uuid
import requests
from keystoneauth1 import exceptions as ks_exc
from keystoneauth1 import loading as keystone
from oslo_log import log as logging
from oslo_serialization import jsonutils
from oslo_utils import versionutils
from neutron_lib._i18n import _
from neutron_lib.exceptions import placement as n_exc
class NoAuthClient(object):
    """Placement NoAuthClient for fullstack testing"""

    def __init__(self, url):
        self.url = url
        self.timeout = 5
        self.retries = 2

    def request(self, url, method, body=None, headers=None, **kwargs):
        headers = headers or {}
        headers.setdefault('Accept', 'application/json')
        body = jsonutils.dumps(body, cls=UUIDEncoder)
        for i in range(self.retries):
            try:
                resp = requests.request(method, url, data=body, headers=headers, verify=False, timeout=self.timeout, **kwargs)
                return resp
            except requests.Timeout:
                LOG.exception("requests Timeout, let's retry it...")
            except requests.ConnectionError:
                LOG.exception('Connection Error appeared')
            except requests.RequestException:
                LOG.exception("Some really weird thing happened, let's retry it")
            time.sleep(self.timeout)
        raise ks_exc.HttpError

    def get(self, url, endpoint_filter, **kwargs):
        return self.request('%s%s' % (self.url, url), 'GET', **kwargs)

    def post(self, url, json, endpoint_filter, **kwargs):
        return self.request('%s%s' % (self.url, url), 'POST', body=json, **kwargs)

    def put(self, url, json, endpoint_filter, **kwargs):
        resp = self.request('%s%s' % (self.url, url), 'PUT', body=json, **kwargs)
        return resp

    def delete(self, url, endpoint_filter, **kwargs):
        return self.request('%s%s' % (self.url, url), 'DELETE', **kwargs)