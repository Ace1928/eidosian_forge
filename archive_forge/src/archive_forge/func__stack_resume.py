import os
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib import exceptions as tempest_exc
def _stack_resume(self, id, wait=True):
    cmd = 'stack resume ' + id
    if wait:
        cmd += ' --wait'
    stack_raw = self.openstack(cmd)
    stack = self.list_to_dict(stack_raw, id)
    return stack