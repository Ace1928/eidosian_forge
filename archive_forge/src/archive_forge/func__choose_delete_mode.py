import sys
from osc_lib.command import command
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from saharaclient.osc import utils
from saharaclient.osc.v1 import clusters as c_v1
def _choose_delete_mode(self, parsed_args):
    if parsed_args.force:
        return 'force_delete'
    else:
        return 'delete'