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
class V3(CommonIdentityTests, utils.TestCase):

    @property
    def version(self):
        return 'v3'

    @property
    def discovery_version(self):
        return '3.0'

    def get_auth_data(self, **kwargs):
        kwargs.setdefault('project_id', self.PROJECT_ID)
        token = fixture.V3Token(**kwargs)
        region = 'RegionOne'
        svc = token.add_service('identity')
        svc.add_standard_endpoints(admin=self.TEST_ADMIN_URL, region=region)
        svc = token.add_service('compute')
        svc.add_standard_endpoints(admin=self.TEST_COMPUTE_ADMIN, public=self.TEST_COMPUTE_PUBLIC, internal=self.TEST_COMPUTE_INTERNAL, region=region)
        svc = token.add_service('volumev2')
        svc.add_standard_endpoints(admin=self.TEST_VOLUME.versions['v2'].service.admin, public=self.TEST_VOLUME.versions['v2'].service.public, internal=self.TEST_VOLUME.versions['v2'].service.internal, region=region)
        svc = token.add_service('volumev3')
        svc.add_standard_endpoints(admin=self.TEST_VOLUME.versions['v3'].service.admin, public=self.TEST_VOLUME.versions['v3'].service.public, internal=self.TEST_VOLUME.versions['v3'].service.internal, region=region)
        svc = token.add_service('block-storage')
        svc.add_standard_endpoints(admin=self.TEST_VOLUME.versions['v3'].service.admin, public=self.TEST_VOLUME.versions['v3'].service.public, internal=self.TEST_VOLUME.versions['v3'].service.internal, region=region)
        svc = token.add_service('baremetal')
        svc.add_standard_endpoints(internal=self.TEST_BAREMETAL_INTERNAL, region=region)
        return token

    def stub_auth(self, subject_token=None, **kwargs):
        if not subject_token:
            subject_token = self.TEST_TOKEN
        kwargs.setdefault('headers', {})['X-Subject-Token'] = subject_token
        self.stub_url('POST', ['auth', 'tokens'], **kwargs)

    def create_auth_plugin(self, **kwargs):
        kwargs.setdefault('auth_url', self.TEST_URL)
        kwargs.setdefault('username', self.TEST_USER)
        kwargs.setdefault('password', self.TEST_PASS)
        return identity.V3Password(**kwargs)