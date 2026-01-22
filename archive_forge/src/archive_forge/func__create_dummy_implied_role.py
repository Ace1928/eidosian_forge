import os
import fixtures
from tempest.lib.common.utils import data_utils
from openstackclient.tests.functional import base
def _create_dummy_implied_role(self, add_clean_up=True):
    role_name = self._create_dummy_role(add_clean_up)
    implied_role_name = self._create_dummy_role(add_clean_up)
    self.openstack('implied role create --implied-role %(implied_role)s %(role)s' % {'implied_role': implied_role_name, 'role': role_name})
    return (implied_role_name, role_name)