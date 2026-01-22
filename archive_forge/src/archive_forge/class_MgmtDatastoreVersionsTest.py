import testtools
from unittest import mock
from troveclient import base
from troveclient.v1 import management
class MgmtDatastoreVersionsTest(testtools.TestCase):

    def setUp(self):
        super(MgmtDatastoreVersionsTest, self).setUp()
        self.orig__init = management.MgmtDatastoreVersions.__init__
        management.MgmtDatastoreVersions.__init__ = mock.Mock(return_value=None)
        self.ds_version = management.MgmtDatastoreVersions()
        self.ds_version.api = mock.Mock()
        self.ds_version.api.client = mock.Mock()
        self.ds_version.resource_class = mock.Mock(return_value='ds-version-1')
        self.orig_base_getid = base.getid
        base.getid = mock.Mock(return_value='ds-version1')

    def tearDown(self):
        super(MgmtDatastoreVersionsTest, self).tearDown()
        management.MgmtDatastoreVersions.__init__ = self.orig__init
        base.getid = self.orig_base_getid

    def _get_mock_method(self):
        self._resp = mock.Mock()
        self._body = None
        self._url = None

        def side_effect_func(url, body=None):
            self._body = body
            self._url = url
            return (self._resp, body)
        return mock.Mock(side_effect=side_effect_func)

    def test_create(self):

        def side_effect_func(path, body, *kw):
            return (path, body)
        self.ds_version._create = mock.Mock(side_effect=side_effect_func)
        p, b = self.ds_version.create('ds-version1', 'mysql', 'mysql', 'image-id', ['mysql-server-5.5'], 'true', 'true')
        self.assertEqual('/mgmt/datastore-versions', p)
        self.assertEqual('ds-version1', b['version']['name'])
        self.assertEqual('mysql', b['version']['datastore_name'])
        self.assertEqual('mysql', b['version']['datastore_manager'])
        self.assertEqual('image-id', b['version']['image'])
        self.assertEqual(['mysql-server-5.5'], b['version']['packages'])
        self.assertTrue(b['version']['active'])
        self.assertTrue(b['version']['default'])

    def test_get(self):

        def side_effect_func(path, ins):
            return (path, ins)
        self.ds_version._get = mock.Mock(side_effect=side_effect_func)
        p, i = self.ds_version.get('ds-version-1')
        self.assertEqual(('/mgmt/datastore-versions/ds-version-1', 'version'), (p, i))

    def test_list(self):
        page_mock = mock.Mock()
        self.ds_version._paginated = page_mock
        self.ds_version.list()
        page_mock.assert_called_with('/mgmt/datastore-versions', 'versions', None, None)
        self.ds_version.list(limit=10, marker='foo')
        page_mock.assert_called_with('/mgmt/datastore-versions', 'versions', 10, 'foo')

    def test_delete(self):
        resp = mock.Mock()
        resp.status_code = 202
        self.ds_version.api.client.delete = mock.Mock(return_value=(resp, None))
        self.ds_version.delete('ds-version-1')
        self.assertEqual(1, self.ds_version.api.client.delete.call_count)
        self.ds_version.api.client.delete.assert_called_with('/mgmt/datastore-versions/ds-version-1')
        resp.status_code = 400
        self.assertRaises(Exception, self.ds_version.delete, 'ds-version-2')
        self.assertEqual(2, self.ds_version.api.client.delete.call_count)
        self.ds_version.api.client.delete.assert_called_with('/mgmt/datastore-versions/ds-version-2')

    def test_edit(self):
        self.ds_version.api.client.patch = self._get_mock_method()
        self._resp.status_code = 202
        self.ds_version.edit('ds-version-1', image='new-image-id')
        self.assertEqual('/mgmt/datastore-versions/ds-version-1', self._url)
        self.assertEqual({'image': 'new-image-id'}, self._body)
        self._resp.status_code = 400
        self.assertRaises(Exception, self.ds_version.edit, 'ds-version-1', 'new-mgr', 'non-existent-image')