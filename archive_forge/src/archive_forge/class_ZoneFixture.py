import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
class ZoneFixture(BaseFixture):
    """See DesignateCLI.zone_create for __init__ args"""

    def _setUp(self):
        super()._setUp()
        self.zone = self.client.zone_create(*self.args, **self.kwargs)
        self.addCleanup(self.cleanup_zone, self.client, self.zone.id)

    @classmethod
    def cleanup_zone(cls, client, zone_id):
        try:
            client.zone_delete(zone_id)
        except CommandFailed:
            pass