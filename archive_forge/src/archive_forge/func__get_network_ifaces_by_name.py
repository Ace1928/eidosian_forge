from oslo_log import log as logging
from os_win import constants
from os_win import exceptions
from os_win.utils import baseutils
from os_win.utils.network import networkutils
def _get_network_ifaces_by_name(self, network_name):
    return [n for n in self._scimv2.MSFT_NetAdapter() if n.Name.find(network_name) >= 0]