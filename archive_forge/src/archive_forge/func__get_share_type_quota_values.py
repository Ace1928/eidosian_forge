from random import randint
import ddt
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def _get_share_type_quota_values(project_quota_value):
    project_quota_value = int(project_quota_value)
    if project_quota_value == -1:
        return randint(1, 999)
    elif project_quota_value == 0:
        return 0
    else:
        return project_quota_value - 1