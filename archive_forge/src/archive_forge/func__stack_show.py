import os
from tempest.lib.cli import base
from tempest.lib.cli import output_parser
from tempest.lib import exceptions as tempest_exc
def _stack_show(self, stack_id):
    cmd = 'stack show ' + stack_id
    stack_raw = self.openstack(cmd)
    return self.show_to_dict(stack_raw)