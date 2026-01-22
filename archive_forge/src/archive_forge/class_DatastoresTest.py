import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import datastores
class DatastoresTest(testtools.TestCase):

    def setUp(self):
        super(DatastoresTest, self).setUp()
        self.orig__init = datastores.Datastores.__init__
        datastores.Datastores.__init__ = mock.Mock(return_value=None)
        self.datastores = datastores.Datastores()
        self.datastores.api = mock.Mock()
        self.datastores.api.client = mock.Mock()
        self.datastores.resource_class = mock.Mock(return_value='ds-1')
        self.orig_base_getid = base.getid
        base.getid = mock.Mock(return_value='datastore1')

    def tearDown(self):
        super(DatastoresTest, self).tearDown()
        datastores.Datastores.__init__ = self.orig__init
        base.getid = self.orig_base_getid

    def test_list(self):
        page_mock = mock.Mock()
        self.datastores._paginated = page_mock
        limit = 'test-limit'
        marker = 'test-marker'
        self.datastores.list(limit, marker)
        page_mock.assert_called_with('/datastores', 'datastores', limit, marker)
        self.datastores.list()
        page_mock.assert_called_with('/datastores', 'datastores', None, None)

    def test_get(self):

        def side_effect_func(path, inst):
            return (path, inst)
        self.datastores._get = mock.Mock(side_effect=side_effect_func)
        self.assertEqual(('/datastores/datastore1', 'datastore'), self.datastores.get(1))