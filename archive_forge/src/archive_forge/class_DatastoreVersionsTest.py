import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import datastores
class DatastoreVersionsTest(testtools.TestCase):

    def setUp(self):
        super(DatastoreVersionsTest, self).setUp()
        self.orig__init = datastores.DatastoreVersions.__init__
        datastores.DatastoreVersions.__init__ = mock.Mock(return_value=None)
        self.datastore_versions = datastores.DatastoreVersions()
        self.datastore_versions.api = mock.Mock()
        self.datastore_versions.api.client = mock.Mock()
        self.datastore_versions.resource_class = mock.Mock(return_value='ds_version-1')
        self.orig_base_getid = base.getid
        base.getid = mock.Mock(return_value='datastore_version1')

    def tearDown(self):
        super(DatastoreVersionsTest, self).tearDown()
        datastores.DatastoreVersions.__init__ = self.orig__init
        base.getid = self.orig_base_getid

    def test_list(self):
        page_mock = mock.Mock()
        self.datastore_versions._paginated = page_mock
        limit = 'test-limit'
        marker = 'test-marker'
        self.datastore_versions.list('datastore1', limit, marker)
        page_mock.assert_called_with('/datastores/datastore1/versions', 'versions', limit, marker)

    def test_get(self):

        def side_effect_func(path, inst):
            return (path, inst)
        self.datastore_versions._get = mock.Mock(side_effect=side_effect_func)
        self.assertEqual(('/datastores/datastore1/versions/datastore_version1', 'version'), self.datastore_versions.get('datastore1', 'datastore_version1'))

    def test_get_by_uuid(self):

        def side_effect_func(path, inst):
            return (path, inst)
        self.datastore_versions._get = mock.Mock(side_effect=side_effect_func)
        self.assertEqual(('/datastores/versions/datastore_version1', 'version'), self.datastore_versions.get_by_uuid('datastore_version1'))