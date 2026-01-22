from random import randint
import ddt
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def _verify_current_st_quotas_equal_to(self, quotas, microversion):
    cmd = 'quota-show --tenant-id %s --share-type %s' % (self.project_id, self.st_id)
    st_quotas_raw = self.admin_client.manila(cmd, microversion=microversion)
    st_quotas = output_parser.details(st_quotas_raw)
    self.assertGreater(len(st_quotas), 3)
    for key, value in st_quotas.items():
        if key not in ('shares', 'gigabytes', 'snapshots', 'snapshot_gigabytes'):
            continue
        self.assertIn(key, quotas)
        self.assertEqual(int(quotas[key]), int(value))