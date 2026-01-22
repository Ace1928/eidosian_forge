import sys
from osc_lib.command import command
from osc_lib import exceptions
from osc_lib import utils as osc_utils
from oslo_log import log as logging
from oslo_serialization import jsonutils as json
from saharaclient.osc import utils
def _format_node_groups_list(node_groups):
    return ', '.join(['%s:%s' % (ng['name'], ng['count']) for ng in node_groups])