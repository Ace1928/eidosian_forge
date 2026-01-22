import testtools
from unittest import mock
from troveclient.compat import auth
from troveclient.compat import exceptions
class FakeAuthTest(testtools.TestCase):

    def test_authenticate(self):
        tenant = 'tenant'
        authObj = auth.FakeAuth(url=None, type=auth.FakeAuth, client=None, username=None, password=None, tenant=tenant)
        fc = authObj.authenticate()
        public_url = '%s/%s' % ('http://localhost:8779/v1.0', tenant)
        self.assertEqual(public_url, fc.get_public_url())
        self.assertEqual(tenant, fc.get_token())