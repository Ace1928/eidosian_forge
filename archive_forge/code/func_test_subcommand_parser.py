from unittest import mock
from urllib import parse
import ddt
import fixtures
from requests_mock.contrib import fixture as requests_mock_fixture
import cinderclient
from cinderclient import api_versions
from cinderclient import base
from cinderclient import client
from cinderclient import exceptions
from cinderclient import shell
from cinderclient.tests.unit.fixture_data import keystone_client
from cinderclient.tests.unit import utils
from cinderclient.tests.unit.v3 import fakes
from cinderclient import utils as cinderclient_utils
from cinderclient.v3 import attachments
from cinderclient.v3 import volume_snapshots
from cinderclient.v3 import volumes
def test_subcommand_parser(self):
    """Ensure that all the expected commands show up.

        This test ensures that refactoring code does not somehow result in
        a command accidentally ceasing to exist.

        TODO: add a similar test for 3.59 or so
        """
    p = self.shell.get_subcommand_parser(api_versions.APIVersion('3.0'), input_args=['help'], do_help=True)
    help_text = p.format_help()
    expected_commands = ('absolute-limits', 'api-version', 'availability-zone-list', 'backup-create', 'backup-delete', 'backup-export', 'backup-import', 'backup-list', 'backup-reset-state', 'backup-restore', 'backup-show', 'cgsnapshot-create', 'cgsnapshot-delete', 'cgsnapshot-list', 'cgsnapshot-show', 'consisgroup-create', 'consisgroup-create-from-src', 'consisgroup-delete', 'consisgroup-list', 'consisgroup-show', 'consisgroup-update', 'create', 'delete', 'encryption-type-create', 'encryption-type-delete', 'encryption-type-list', 'encryption-type-show', 'encryption-type-update', 'extend', 'extra-specs-list', 'failover-host', 'force-delete', 'freeze-host', 'get-capabilities', 'get-pools', 'image-metadata', 'image-metadata-show', 'list', 'manage', 'metadata', 'metadata-show', 'metadata-update-all', 'migrate', 'qos-associate', 'qos-create', 'qos-delete', 'qos-disassociate', 'qos-disassociate-all', 'qos-get-association', 'qos-key', 'qos-list', 'qos-show', 'quota-class-show', 'quota-class-update', 'quota-defaults', 'quota-delete', 'quota-show', 'quota-update', 'quota-usage', 'rate-limits', 'readonly-mode-update', 'rename', 'reset-state', 'retype', 'service-disable', 'service-enable', 'service-list', 'set-bootable', 'show', 'snapshot-create', 'snapshot-delete', 'snapshot-list', 'snapshot-manage', 'snapshot-metadata', 'snapshot-metadata-show', 'snapshot-metadata-update-all', 'snapshot-rename', 'snapshot-reset-state', 'snapshot-show', 'snapshot-unmanage', 'thaw-host', 'transfer-accept', 'transfer-create', 'transfer-delete', 'transfer-list', 'transfer-show', 'type-access-add', 'type-access-list', 'type-access-remove', 'type-create', 'type-default', 'type-delete', 'type-key', 'type-list', 'type-show', 'type-update', 'unmanage', 'upload-to-image', 'version-list', 'bash-completion', 'help')
    for e in expected_commands:
        self.assertIn('    ' + e, help_text)