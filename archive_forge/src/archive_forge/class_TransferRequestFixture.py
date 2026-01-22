import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
class TransferRequestFixture(BaseFixture):
    """See DesignateCLI.zone_transfer_request_create for __init__ args"""

    def __init__(self, zone, user='default', target_user='alt', *args, **kwargs):
        super().__init__(user, *args, **kwargs)
        self.zone = zone
        self.target_client = DesignateCLI.as_user(target_user)
        self.kwargs['target_project_id'] = self.target_client.project_id

    def _setUp(self):
        super()._setUp()
        self.transfer_request = self.client.zone_transfer_request_create(*self.args, zone_id=self.zone.id, **self.kwargs)
        self.addCleanup(self.cleanup_transfer_request, self.client, self.transfer_request.id)
        self.addCleanup(ZoneFixture.cleanup_zone, self.client, self.zone.id)
        self.addCleanup(ZoneFixture.cleanup_zone, self.target_client, self.zone.id)

    @classmethod
    def cleanup_transfer_request(cls, client, transfer_request_id):
        try:
            client.zone_transfer_request_delete(transfer_request_id)
        except CommandFailed:
            pass