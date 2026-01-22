from unittest import mock
from unittest.mock import call
from cinderclient import api_versions
from osc_lib.cli import format_columns
from osc_lib import exceptions
from osc_lib import utils
from openstackclient.tests.unit.identity.v3 import fakes as identity_fakes
from openstackclient.tests.unit import utils as tests_utils
from openstackclient.tests.unit.volume.v2 import fakes as volume_fakes
from openstackclient.volume.v2 import volume_type
class TestColumns(TestType):

    def test_encryption_info_column_with_info(self):
        fake_volume_type = volume_fakes.create_one_volume_type()
        type_id = fake_volume_type.id
        encryption_info = {'provider': 'LuksEncryptor', 'cipher': None, 'key_size': None, 'control_location': 'front-end'}
        col = volume_type.EncryptionInfoColumn(type_id, {type_id: encryption_info})
        self.assertEqual(utils.format_dict(encryption_info), col.human_readable())
        self.assertEqual(encryption_info, col.machine_readable())

    def test_encryption_info_column_without_info(self):
        fake_volume_type = volume_fakes.create_one_volume_type()
        type_id = fake_volume_type.id
        col = volume_type.EncryptionInfoColumn(type_id, {})
        self.assertEqual('-', col.human_readable())
        self.assertIsNone(col.machine_readable())