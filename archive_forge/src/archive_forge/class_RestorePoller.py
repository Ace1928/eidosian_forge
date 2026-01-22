from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
class RestorePoller(object):
    """Restore poller for polling restore until it's terminal."""

    def __init__(self, client, messages):
        self.client = client
        self.messages = messages

    def IsNotDone(self, restore, unused_state):
        del unused_state
        return not (restore.state == self.messages.Restore.StateValueValuesEnum.SUCCEEDED or restore.state == self.messages.Restore.StateValueValuesEnum.FAILED or restore.state == self.messages.Restore.StateValueValuesEnum.DELETING)

    def _GetRestore(self, restore):
        req = self.messages.GkebackupProjectsLocationsRestorePlansRestoresGetRequest()
        req.name = restore
        return self.client.projects_locations_restorePlans_restores.Get(req)

    def Poll(self, restore):
        return self._GetRestore(restore)

    def GetResult(self, restore):
        return self._GetRestore(restore)