import os
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib import exceptions as tempest_exc
def _stack_snapshot_create(self, id, name):
    cmd = 'stack snapshot create ' + id + ' --name ' + name
    snapshot_raw = self.openstack(cmd)
    snapshot = self.show_to_dict(snapshot_raw)
    self.addCleanup(self._stack_snapshot_delete, id, snapshot['id'])
    return snapshot