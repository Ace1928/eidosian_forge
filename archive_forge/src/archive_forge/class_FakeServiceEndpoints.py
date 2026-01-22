import abc
import collections
import urllib
import uuid
from keystoneauth1 import _utils
from keystoneauth1 import access
from keystoneauth1 import adapter
from keystoneauth1 import discover
from keystoneauth1 import exceptions
from keystoneauth1 import fixture
from keystoneauth1 import identity
from keystoneauth1 import plugin
from keystoneauth1 import session
from keystoneauth1.tests.unit import utils
class FakeServiceEndpoints(object):

    def __init__(self, base_url, versions=None, project_id=None, **kwargs):
        self.base_url = base_url
        self._interfaces = {}
        for interface in ('public', 'internal', 'admin'):
            if interface in kwargs and (not kwargs[interface]):
                self._interfaces[interface] = False
            else:
                self._interfaces[interface] = True
        self.versions = {}
        self.unversioned = self._make_urls()
        if not versions:
            self.catalog = self.unversioned
        else:
            self.catalog = self._make_urls(versions[0], project_id)
            for version in versions:
                self.versions[version] = _ServiceVersion(self._make_urls(version), self._make_urls(version, project_id))

    def _make_urls(self, *parts):
        return _Endpoints(self._make_url('public', *parts), self._make_url('internal', *parts), self._make_url('admin', *parts))

    def _make_url(self, interface, *parts):
        if not self._interfaces[interface]:
            return None
        url = urllib.parse.urljoin(self.base_url + '/', interface)
        for part in parts:
            if part:
                url = urllib.parse.urljoin(url + '/', part)
        return url