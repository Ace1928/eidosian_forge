import sys
import tempfile
import traceback
import fixtures
from tempest.lib.exceptions import CommandFailed
from testtools.runtest import MultipleExceptions
from designateclient.functionaltests.client import DesignateCLI
class TSIGKeyFixture(BaseFixture):
    """See DesignateCLI.tsigkey_create for __init__ args"""

    def __init__(self, user='admin', *args, **kwargs):
        super().__init__(*args, user=user, **kwargs)

    def _setUp(self):
        super()._setUp()
        self.tsigkey = self.client.tsigkey_create(*self.args, **self.kwargs)
        self.addCleanup(self.cleanup_tsigkey(self.client, self.tsigkey.id))

    @classmethod
    def cleanup_tsigkey(cls, client, tsigkey_id):
        try:
            client.tsigkey_delete(tsigkey_id)
        except CommandFailed:
            pass