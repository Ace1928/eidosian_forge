from unittest import mock
from blazarclient import base
from blazarclient import exception
from blazarclient import tests
class BaseClientManagerTestCase(tests.TestCase):

    def setUp(self):
        super(BaseClientManagerTestCase, self).setUp()
        self.blazar_url = 'www.fake.com/reservation'
        self.auth_token = 'aaa-bbb-ccc'
        self.session = mock.MagicMock()
        self.user_agent = 'python-blazarclient'

    def test_init_with_session(self):
        manager = base.BaseClientManager(blazar_url=None, auth_token=None, session=self.session)
        self.assertIsInstance(manager.request_manager, base.SessionClient)

    def test_init_with_url_and_token(self):
        manager = base.BaseClientManager(blazar_url=self.blazar_url, auth_token=self.auth_token, session=None)
        self.assertIsInstance(manager.request_manager, base.RequestManager)

    def test_init_with_insufficient_info(self):
        self.assertRaises(exception.InsufficientAuthInformation, base.BaseClientManager, blazar_url=None, auth_token=self.auth_token, session=None)