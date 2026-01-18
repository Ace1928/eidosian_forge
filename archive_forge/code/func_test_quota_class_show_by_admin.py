from random import randint
import ddt
from tempest.lib.cli import output_parser
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions
from manilaclient import api_versions
from manilaclient.tests.functional import base
from manilaclient.tests.functional import utils
def test_quota_class_show_by_admin(self):
    roles = self.parser.listing(self.clients['admin'].manila('quota-class-show', params='abc'))
    self.assertTableStruct(roles, ('Property', 'Value'))