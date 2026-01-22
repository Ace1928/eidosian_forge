import os
import unittest
import fixtures
from tempest.lib.common.utils import data_utils
from tempest.lib import exceptions as tempest_exceptions
from openstackclient.tests.functional import base
def _create_dummy_ec2_credentials(self, add_clean_up=True):
    raw_output = self.openstack('ec2 credentials create')
    ec2_credentials = self.parse_show_as_object(raw_output)
    access_key = ec2_credentials['access']
    if add_clean_up:
        self.addCleanup(self.openstack, 'ec2 credentials delete %s' % access_key)
    items = self.parse_show(raw_output)
    self.assert_show_fields(items, self.EC2_CREDENTIALS_FIELDS)
    return access_key