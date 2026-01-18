import string
from unittest import mock
import netaddr
from neutron_lib._i18n import _
from neutron_lib.api import converters
from neutron_lib.api.definitions import extra_dhcp_opt
from neutron_lib.api import validators
from neutron_lib import constants
from neutron_lib import exceptions as n_exc
from neutron_lib import fixture
from neutron_lib.plugins import directory
from neutron_lib.tests import _base as base
def test_validate_uuid(self):
    invalid_uuids = [None, 123, '123', 't5069610-744b-42a7-8bd8-ceac1a229cd4', 'e5069610-744bb-42a7-8bd8-ceac1a229cd4']
    for uuid in invalid_uuids:
        msg = validators.validate_uuid(uuid)
        error = "'%s' is not a valid UUID" % uuid
        self.assertEqual(error, msg)
    msg = validators.validate_uuid('00000000-ffff-ffff-ffff-000000000000')
    self.assertIsNone(msg)