import functools
import re
from eventlet import patcher
from eventlet import tpool
from oslo_log import log as logging
from oslo_utils import units
import six
from os_win._i18n import _
from os_win import conf
from os_win import constants
from os_win import exceptions
from os_win.utils import _wqlutils
from os_win.utils import baseutils
from os_win.utils import jobutils
def _get_vswitch_external_port(self, vswitch_name):
    vswitch = self._get_vswitch(vswitch_name)
    ext_ports = self._conn.Msvm_ExternalEthernetPort()
    for ext_port in ext_ports:
        lan_endpoint_assoc_list = self._conn.Msvm_EthernetDeviceSAPImplementation(Antecedent=ext_port.path_())
        if lan_endpoint_assoc_list:
            lan_endpoint_assoc_list = self._conn.Msvm_ActiveConnection(Dependent=lan_endpoint_assoc_list[0].Dependent.path_())
            if lan_endpoint_assoc_list:
                lan_endpoint = lan_endpoint_assoc_list[0].Antecedent
                if lan_endpoint.SystemName == vswitch.Name:
                    return ext_port