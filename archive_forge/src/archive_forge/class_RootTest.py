import testtools
from unittest import mock
from troveclient.v1 import root
class RootTest(testtools.TestCase):

    def setUp(self):
        super(RootTest, self).setUp()
        self.orig__init = root.Root.__init__
        root.Root.__init__ = mock.Mock(return_value=None)
        self.root = root.Root()
        self.root.api = mock.Mock()
        self.root.api.client = mock.Mock()

    def tearDown(self):
        super(RootTest, self).tearDown()
        root.Root.__init__ = self.orig__init

    def _get_mock_method(self):
        self._resp = mock.Mock()
        self._body = None
        self._url = None

        def side_effect_func(url, body=None):
            self._body = body
            self._url = url
            return (self._resp, body)
        return mock.Mock(side_effect=side_effect_func)

    def test_delete(self):
        self.root.api.client.delete = self._get_mock_method()
        self._resp.status_code = 200
        self.root.delete(1234)
        self.assertEqual('/instances/1234/root', self._url)
        self._resp.status_code = 400
        self.assertRaises(Exception, self.root.delete, 1234)