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
class GtmXPoolsParameters(BaseParameters):
    api_map = {'alternateMode': 'alternate_mode', 'dynamicRatio': 'dynamic_ratio', 'fallbackMode': 'fallback_mode', 'fullPath': 'full_path', 'loadBalancingMode': 'load_balancing_mode', 'manualResume': 'manual_resume', 'maxAnswersReturned': 'max_answers_returned', 'qosHitRatio': 'qos_hit_ratio', 'qosHops': 'qos_hops', 'qosKilobytesSecond': 'qos_kilobytes_second', 'qosLcs': 'qos_lcs', 'qosPacketRate': 'qos_packet_rate', 'qosRtt': 'qos_rtt', 'qosTopology': 'qos_topology', 'qosVsCapacity': 'qos_vs_capacity', 'qosVsScore': 'qos_vs_score', 'verifyMemberAvailability': 'verify_member_availability', 'membersReference': 'members'}
    returnables = ['alternate_mode', 'dynamic_ratio', 'enabled', 'disabled', 'fallback_mode', 'full_path', 'load_balancing_mode', 'manual_resume', 'max_answers_returned', 'members', 'name', 'partition', 'qos_hit_ratio', 'qos_hops', 'qos_kilobytes_second', 'qos_lcs', 'qos_packet_rate', 'qos_rtt', 'qos_topology', 'qos_vs_capacity', 'qos_vs_score', 'ttl', 'verify_member_availability']

    @property
    def verify_member_availability(self):
        return flatten_boolean(self._values['verify_member_availability'])

    @property
    def dynamic_ratio(self):
        return flatten_boolean(self._values['dynamic_ratio'])

    @property
    def max_answers_returned(self):
        if self._values['max_answers_returned'] is None:
            return None
        return int(self._values['max_answers_returned'])

    @property
    def members(self):
        result = []
        if self._values['members'] is None or 'items' not in self._values['members']:
            return result
        for item in self._values['members']['items']:
            self._remove_internal_keywords(item)
            if 'disabled' in item:
                item['disabled'] = flatten_boolean(item['disabled'])
                item['enabled'] = flatten_boolean(not item['disabled'])
            if 'enabled' in item:
                item['enabled'] = flatten_boolean(item['enabled'])
                item['disabled'] = flatten_boolean(not item['enabled'])
            if 'fullPath' in item:
                item['full_path'] = item.pop('fullPath')
            if 'memberOrder' in item:
                item['member_order'] = int(item.pop('memberOrder'))
            for x in ['order', 'preference', 'ratio', 'service']:
                if x in item:
                    item[x] = int(item[x])
            result.append(item)
        return result

    @property
    def qos_hit_ratio(self):
        if self._values['qos_hit_ratio'] is None:
            return None
        return int(self._values['qos_hit_ratio'])

    @property
    def qos_hops(self):
        if self._values['qos_hops'] is None:
            return None
        return int(self._values['qos_hops'])

    @property
    def qos_kilobytes_second(self):
        if self._values['qos_kilobytes_second'] is None:
            return None
        return int(self._values['qos_kilobytes_second'])

    @property
    def qos_lcs(self):
        if self._values['qos_lcs'] is None:
            return None
        return int(self._values['qos_lcs'])

    @property
    def qos_packet_rate(self):
        if self._values['qos_packet_rate'] is None:
            return None
        return int(self._values['qos_packet_rate'])

    @property
    def qos_rtt(self):
        if self._values['qos_rtt'] is None:
            return None
        return int(self._values['qos_rtt'])

    @property
    def qos_topology(self):
        if self._values['qos_topology'] is None:
            return None
        return int(self._values['qos_topology'])

    @property
    def qos_vs_capacity(self):
        if self._values['qos_vs_capacity'] is None:
            return None
        return int(self._values['qos_vs_capacity'])

    @property
    def qos_vs_score(self):
        if self._values['qos_vs_score'] is None:
            return None
        return int(self._values['qos_vs_score'])

    @property
    def availability_state(self):
        if self._values['stats'] is None:
            return None
        try:
            result = self._values['stats']['status']['availabilityState']
            return result['description']
        except AttributeError:
            return None

    @property
    def enabled_state(self):
        if self._values['stats'] is None:
            return None
        try:
            result = self._values['stats']['status']['enabledState']
            return result['description']
        except AttributeError:
            return None

    @property
    def availability_status(self):
        if self.enabled_state == 'enabled':
            if self.availability_state == 'offline':
                return 'red'
            elif self.availability_state == 'available':
                return 'green'
            elif self.availability_state == 'unknown':
                return 'blue'
            else:
                return 'none'
        else:
            return 'black'

    @property
    def manual_resume(self):
        return flatten_boolean(self._values['manual_resume'])