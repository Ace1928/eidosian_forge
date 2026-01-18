import abc
import netaddr
from oslo_db import exception
from oslo_db.sqlalchemy import enginefacade
from oslo_db.sqlalchemy import test_fixtures
from oslo_utils import timeutils
from oslo_utils import uuidutils
import sqlalchemy as sa
from neutron_lib import context
from neutron_lib.db import sqlalchemytypes
from neutron_lib.tests import _base as test_base
from neutron_lib.tests import tools
from neutron_lib.utils import net
def test_wrong_mac(self):
    wrong_macs = ['fake', '', -1, 'FK:16:3E:00:00:02', 'FA:16:3E:00:00:020']
    for mac in wrong_macs:
        self.assertRaises(exception.DBError, self._add_row, id=uuidutils.generate_uuid(), mac=mac)