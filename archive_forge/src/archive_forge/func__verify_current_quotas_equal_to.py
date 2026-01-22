from random import randint
import ddt
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def _verify_current_quotas_equal_to(self, quotas, microversion):
    cmd = 'quota-show --tenant-id %s' % self.project_id
    quotas_raw = self.admin_client.manila(cmd, microversion=microversion)
    quotas = output_parser.details(quotas_raw)
    self.assertGreater(len(quotas), 3)
    for key, value in quotas.items():
        if key not in ('shares', 'gigabytes', 'snapshots', 'snapshot_gigabytes', 'share_groups', 'share_group_snapshots'):
            continue
        self.assertIn(key, quotas)
        self.assertEqual(int(quotas[key]), int(value))