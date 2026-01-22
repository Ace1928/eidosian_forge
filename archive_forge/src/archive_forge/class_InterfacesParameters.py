from __future__ import absolute_import, division, print_function
import datetime
import math
import re
import time
import traceback
from collections import namedtuple
from ansible.module_utils.basic import (
from ansible.module_utils.parsing.convert_bool import BOOLEANS_TRUE
from ansible.module_utils.six import (
from ansible.module_utils.urls import urlparse
from ipaddress import ip_interface
from ..module_utils.bigip import F5RestClient
from ..module_utils.common import (
from ..module_utils.urls import parseStats
from ..module_utils.icontrol import (
from ..module_utils.ipaddress import is_valid_ip
from ..module_utils.teem import send_teem
class InterfacesParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'mediaActive': 'active_media_type', 'flowControl': 'flow_control', 'bundleSpeed': 'bundle_speed', 'ifIndex': 'if_index', 'macAddress': 'mac_address', 'mediaSfp': 'media_sfp', 'lldpAdmin': 'lldp_admin', 'preferPort': 'prefer_port', 'stpAutoEdgePort': 'stp_auto_edge_port', 'stp': 'stp_enabled', 'stpLinkType': 'stp_link_type'}
    returnables = ['full_path', 'name', 'active_media_type', 'flow_control', 'description', 'bundle', 'bundle_speed', 'enabled', 'if_index', 'mac_address', 'media_sfp', 'lldp_admin', 'mtu', 'prefer_port', 'sflow_poll_interval', 'sflow_poll_interval_global', 'stp_auto_edge_port', 'stp_enabled', 'stp_link_type']

    @property
    def stp_auto_edge_port(self):
        return flatten_boolean(self._values['stp_auto_edge_port'])

    @property
    def stp_enabled(self):
        return flatten_boolean(self._values['stp_enabled'])

    @property
    def sflow_poll_interval_global(self):
        if self._values['sflow'] is None:
            return None
        if 'pollIntervalGlobal' in self._values['sflow']:
            return self._values['sflow']['pollIntervalGlobal']

    @property
    def sflow_poll_interval(self):
        if self._values['sflow'] is None:
            return None
        if 'pollInterval' in self._values['sflow']:
            return self._values['sflow']['pollInterval']

    @property
    def mac_address(self):
        if self._values['mac_address'] in [None, 'none']:
            return None
        return self._values['mac_address']

    @property
    def enabled(self):
        return flatten_boolean(self._values['enabled'])