import copy
import io
import tempfile
from unittest import mock
from cinderclient import api_versions
from openstack import exceptions as sdk_exceptions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from openstackclient.image.v2 import image as _image
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit.image.v2 import fakes as image_fakes
from openstackclient.tests.unit.volume.v3 import fakes as volume_fakes
class TestStoresInfo(TestImage):
    stores_info = image_fakes.create_one_stores_info()

    def setUp(self):
        super().setUp()
        self.image_client.stores.return_value = self.stores_info
        self.cmd = _image.StoresInfo(self.app, None)

    def test_stores_info(self):
        arglist = []
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.image_client.stores.assert_called()

    def test_stores_info_with_detail(self):
        arglist = ['--detail']
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.cmd.take_action(parsed_args)
        self.image_client.stores.assert_called_with(details=True)

    def test_stores_info_neg(self):
        arglist = []
        parsed_args = self.check_parser(self.cmd, arglist, [])
        self.image_client.stores.side_effect = sdk_exceptions.ResourceNotFound()
        exc = self.assertRaises(exceptions.CommandError, self.cmd.take_action, parsed_args)
        self.assertIn('Multi Backend support not enabled', str(exc))