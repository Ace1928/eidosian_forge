from unittest import mock
import ddt
from manilaclient import api_versions
from manilaclient import extension
from manilaclient.tests.unit import utils
from manilaclient.tests.unit.v2 import fakes
from manilaclient.v2 import share_snapshots
@ddt.data(type('SnapshotUUID', (object,), {'uuid': '1234'}), type('SnapshotID', (object,), {'id': '1234'}), '1234')
def test_update_share_snapshot(self, snapshot):
    data = dict(foo='bar', quuz='foobar')
    snapshot = cs.share_snapshots.update(snapshot, **data)
    cs.assert_called('PUT', '/snapshots/1234', {'snapshot': data})