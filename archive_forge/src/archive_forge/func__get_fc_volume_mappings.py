import collections
import time
from os_win import exceptions as os_win_exc
from os_win import utilsfactory
from oslo_log import log as logging
from os_brick import exception
from os_brick.i18n import _
from os_brick.initiator.windows import base as win_conn_base
from os_brick import utils
def _get_fc_volume_mappings(self, connection_properties):
    target_wwpns = [wwpn.upper() for wwpn in connection_properties['target_wwn']]
    target_lun = connection_properties['target_lun']
    volume_mappings = []
    hba_mappings = self._get_fc_hba_mappings()
    for node_name in hba_mappings:
        target_mappings = self._fc_utils.get_fc_target_mappings(node_name)
        for mapping in target_mappings:
            if mapping['port_name'] in target_wwpns and mapping['lun'] == target_lun:
                volume_mappings.append(mapping)
    return volume_mappings