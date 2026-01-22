from unittest import mock
from novaclient import api_versions
from novaclient import base
from novaclient import exceptions
from novaclient.tests.unit import utils
from novaclient.tests.unit.v2 import fakes
from novaclient.v2 import flavors
class FlavorsTest_v2_55(utils.TestCase):
    """Tests creating/showing/updating a flavor with a description."""

    def setUp(self):
        super(FlavorsTest_v2_55, self).setUp()
        self.cs = fakes.FakeClient(api_versions.APIVersion('2.55'))

    def test_list_flavors(self):
        fl = self.cs.flavors.list()
        self.cs.assert_called('GET', '/flavors/detail')
        for flavor in fl:
            self.assertTrue(hasattr(flavor, 'description'), '%s does not have a description set.' % flavor)

    def test_list_flavors_undetailed(self):
        fl = self.cs.flavors.list(detailed=False)
        self.cs.assert_called('GET', '/flavors')
        for flavor in fl:
            self.assertTrue(hasattr(flavor, 'description'), '%s does not have a description set.' % flavor)

    def test_get_flavor_details(self):
        f = self.cs.flavors.get('with-description')
        self.cs.assert_called('GET', '/flavors/with-description')
        self.assertEqual('test description', f.description)

    def test_create(self):
        self.cs.flavors.create('with-description', 512, 1, 10, 'with-description', ephemeral=10, is_public=False, description='test description')
        body = FlavorsTest._create_body('with-description', 512, 1, 10, 10, 'with-description', 0, 1.0, False)
        body['flavor']['description'] = 'test description'
        self.cs.assert_called('POST', '/flavors', body)

    def test_create_bad_version(self):
        """Tests trying to create a flavor with a description before 2.55."""
        self.cs.api_version = api_versions.APIVersion('2.54')
        self.assertRaises(exceptions.UnsupportedAttribute, self.cs.flavors.create, 'with-description', 512, 1, 10, 'with-description', description='test description')

    def test_update(self):
        updated_flavor = self.cs.flavors.update('with-description', 'new description')
        body = {'flavor': {'description': 'new description'}}
        self.cs.assert_called('PUT', '/flavors/with-description', body)
        self.assertEqual('new description', updated_flavor.description)

    def test_update_bad_version(self):
        """Tests trying to update a flavor with a description before 2.55."""
        self.cs.api_version = api_versions.APIVersion('2.54')
        self.assertRaises(exceptions.VersionNotFoundForAPIMethod, self.cs.flavors.update, 'foo', 'bar')