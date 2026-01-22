import collections
import time
from os_win import exceptions as os_win_exc
from os_win import utilsfactory
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.windows import base as win_conn_base
from os_brick import utils
def _get_fc_hba_mappings(self):
    mappings = collections.defaultdict(list)
    fc_hba_ports = self._fc_utils.get_fc_hba_ports()
    for port in fc_hba_ports:
        mappings[port['node_name']].append(port['port_name'])
    return mappings