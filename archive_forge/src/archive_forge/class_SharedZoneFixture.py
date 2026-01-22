import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
class SharedZoneFixture(BaseFixture):
    """See DesignateCLI.recordset_create for __init__ args"""

    def __init__(self, zone, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.zone = zone

    def _setUp(self):
        super()._setUp()
        self.zone_share = self.client.zone_share(*self.args, zone_id=self.zone.id, **self.kwargs)
        self.addCleanup(self.cleanup_shared_zone, self.client, self.zone.id, self.zone_share.id)

    @classmethod
    def cleanup_shared_zone(cls, client, zone_id, shared_zone_id):
        try:
            client.unshare_zone(zone_id, shared_zone_id)
        except CommandFailed:
            pass