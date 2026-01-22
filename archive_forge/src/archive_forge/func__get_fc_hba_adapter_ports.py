import contextlib
import ctypes
from oslo_log import log as logging
import six
from os_win._i18n import _
from os_win import _utils
import os_win.conf
from os_win import exceptions
from os_win.utils.storage import diskutils
from os_win.utils import win32utils
from os_win.utils.winapi import constants as w_const
from os_win.utils.winapi import libs as w_lib
from os_win.utils.winapi.libs import hbaapi as fc_struct
def _get_fc_hba_adapter_ports(self, adapter_name):
    hba_ports = []
    with self._get_hba_handle(adapter_name=adapter_name) as hba_handle:
        adapter_attributes = self._get_adapter_attributes(hba_handle)
        port_count = adapter_attributes.NumberOfPorts
        for port_index in range(port_count):
            port_attr = self._get_adapter_port_attributes(hba_handle, port_index)
            wwnn = _utils.byte_array_to_hex_str(port_attr.NodeWWN.wwn)
            wwpn = _utils.byte_array_to_hex_str(port_attr.PortWWN.wwn)
            hba_port_info = dict(node_name=wwnn, port_name=wwpn)
            hba_ports.append(hba_port_info)
    return hba_ports