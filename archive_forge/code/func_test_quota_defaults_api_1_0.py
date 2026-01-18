from random import randint
import ddt
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
@ddt.data('admin', 'user')
@utils.skip_if_microversion_not_supported('1.0')
def test_quota_defaults_api_1_0(self, role):
    self._get_quotas(role, 'defaults', '1.0')