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
class MACAddressTestCase(SqlAlchemyTypesBaseTestCase):

    def _get_test_table(self, meta):
        return sa.Table('fakemacaddressmodels', meta, sa.Column('id', sa.String(36), primary_key=True, nullable=False), sa.Column('mac', sqlalchemytypes.MACAddress))

    def _get_one(self, value):
        row_select = self.test_table.select().where(self.test_table.c.mac == value)
        with self.engine.connect() as conn, conn.begin():
            return conn.execute(row_select).first()

    def _get_all(self):
        rows_select = self.test_table.select()
        with self.engine.connect() as conn, conn.begin():
            return conn.execute(rows_select).fetchall()

    def _update_row(self, key, mac):
        row_update = self.test_table.update().values(mac=mac).where(self.test_table.c.mac == key)
        with self.engine.connect() as conn, conn.begin():
            conn.execute(row_update)

    def _delete_row(self):
        row_delete = self.test_table.delete()
        with self.engine.connect() as conn, conn.begin():
            conn.execute(row_delete)

    def test_crud(self):
        mac_addresses = ['FA:16:3E:00:00:01', 'FA:16:3E:00:00:02']
        for mac in mac_addresses:
            mac = netaddr.EUI(mac)
            self._add_row(id=uuidutils.generate_uuid(), mac=mac)
            obj = self._get_one(mac)
            self.assertEqual(mac, obj.mac)
            random_mac = netaddr.EUI(net.get_random_mac(['fe', '16', '3e', '00', '00', '00']))
            self._update_row(mac, random_mac)
            obj = self._get_one(random_mac)
            self.assertEqual(random_mac, obj.mac)
        objs = self._get_all()
        self.assertEqual(len(mac_addresses), len(objs))
        self._delete_rows()
        objs = self._get_all()
        self.assertEqual(0, len(objs))

    def test_wrong_mac(self):
        wrong_macs = ['fake', '', -1, 'FK:16:3E:00:00:02', 'FA:16:3E:00:00:020']
        for mac in wrong_macs:
            self.assertRaises(exception.DBError, self._add_row, id=uuidutils.generate_uuid(), mac=mac)