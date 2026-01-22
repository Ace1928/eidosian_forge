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
class LtmPoolsParameters(BaseParameters):
    api_map = {'fullPath': 'full_path', 'allowNat': 'allow_nat', 'allowSnat': 'allow_snat', 'ignorePersistedWeight': 'ignore_persisted_weight', 'ipTosToClient': 'client_ip_tos', 'ipTosToServer': 'server_ip_tos', 'linkQosToClient': 'client_link_qos', 'linkQosToServer': 'server_link_qos', 'loadBalancingMode': 'lb_method', 'minActiveMembers': 'minimum_active_members', 'minUpMembers': 'minimum_up_members', 'minUpMembersAction': 'minimum_up_members_action', 'minUpMembersChecking': 'minimum_up_members_checking', 'queueDepthLimit': 'queue_depth_limit', 'queueOnConnectionLimit': 'queue_on_connection_limit', 'queueTimeLimit': 'queue_time_limit', 'reselectTries': 'reselect_tries', 'serviceDownAction': 'service_down_action', 'slowRampTime': 'slow_ramp_time', 'monitor': 'monitors'}
    returnables = ['full_path', 'name', 'allow_nat', 'allow_snat', 'description', 'ignore_persisted_weight', 'client_ip_tos', 'server_ip_tos', 'client_link_qos', 'server_link_qos', 'lb_method', 'minimum_active_members', 'minimum_up_members', 'minimum_up_members_action', 'minimum_up_members_checking', 'monitors', 'queue_depth_limit', 'queue_on_connection_limit', 'queue_time_limit', 'reselect_tries', 'service_down_action', 'slow_ramp_time', 'priority_group_activation', 'members', 'metadata', 'active_member_count', 'available_member_count', 'availability_status', 'enabled_status', 'status_reason', 'all_max_queue_entry_age_ever', 'all_avg_queue_entry_age', 'all_queue_head_entry_age', 'all_max_queue_entry_age_recently', 'all_num_connections_queued_now', 'all_num_connections_serviced', 'pool_max_queue_entry_age_ever', 'pool_avg_queue_entry_age', 'pool_queue_head_entry_age', 'pool_max_queue_entry_age_recently', 'pool_num_connections_queued_now', 'pool_num_connections_serviced', 'current_sessions', 'member_count', 'total_requests', 'server_side_bits_in', 'server_side_bits_out', 'server_side_current_connections', 'server_side_max_connections', 'server_side_pkts_in', 'server_side_pkts_out', 'server_side_total_connections']

    @property
    def active_member_count(self):
        if 'availableMemberCnt' in self._values['stats']:
            return int(self._values['stats']['activeMemberCnt'])
        return None

    @property
    def available_member_count(self):
        if 'availableMemberCnt' in self._values['stats']:
            return int(self._values['stats']['availableMemberCnt'])
        return None

    @property
    def all_max_queue_entry_age_ever(self):
        return self._values['stats']['connqAll']['ageEdm']

    @property
    def all_avg_queue_entry_age(self):
        return self._values['stats']['connqAll']['ageEma']

    @property
    def all_queue_head_entry_age(self):
        return self._values['stats']['connqAll']['ageHead']

    @property
    def all_max_queue_entry_age_recently(self):
        return self._values['stats']['connqAll']['ageMax']

    @property
    def all_num_connections_queued_now(self):
        return self._values['stats']['connqAll']['depth']

    @property
    def all_num_connections_serviced(self):
        return self._values['stats']['connqAll']['serviced']

    @property
    def availability_status(self):
        return self._values['stats']['status']['availabilityState']

    @property
    def enabled_status(self):
        return self._values['stats']['status']['enabledState']

    @property
    def status_reason(self):
        return self._values['stats']['status']['statusReason']

    @property
    def pool_max_queue_entry_age_ever(self):
        return self._values['stats']['connq']['ageEdm']

    @property
    def pool_avg_queue_entry_age(self):
        return self._values['stats']['connq']['ageEma']

    @property
    def pool_queue_head_entry_age(self):
        return self._values['stats']['connq']['ageHead']

    @property
    def pool_max_queue_entry_age_recently(self):
        return self._values['stats']['connq']['ageMax']

    @property
    def pool_num_connections_queued_now(self):
        return self._values['stats']['connq']['depth']

    @property
    def pool_num_connections_serviced(self):
        return self._values['stats']['connq']['serviced']

    @property
    def current_sessions(self):
        return self._values['stats']['curSessions']

    @property
    def member_count(self):
        if 'memberCnt' in self._values['stats']:
            return self._values['stats']['memberCnt']
        return None

    @property
    def total_requests(self):
        return self._values['stats']['totRequests']

    @property
    def server_side_bits_in(self):
        return self._values['stats']['serverside']['bitsIn']

    @property
    def server_side_bits_out(self):
        return self._values['stats']['serverside']['bitsOut']

    @property
    def server_side_current_connections(self):
        return self._values['stats']['serverside']['curConns']

    @property
    def server_side_max_connections(self):
        return self._values['stats']['serverside']['maxConns']

    @property
    def server_side_pkts_in(self):
        return self._values['stats']['serverside']['pktsIn']

    @property
    def server_side_pkts_out(self):
        return self._values['stats']['serverside']['pktsOut']

    @property
    def server_side_total_connections(self):
        return self._values['stats']['serverside']['totConns']

    @property
    def ignore_persisted_weight(self):
        return flatten_boolean(self._values['ignore_persisted_weight'])

    @property
    def minimum_up_members_checking(self):
        return flatten_boolean(self._values['minimum_up_members_checking'])

    @property
    def queue_on_connection_limit(self):
        return flatten_boolean(self._values['queue_on_connection_limit'])

    @property
    def priority_group_activation(self):
        """Returns the TMUI value for "Priority Group Activation"

        This value is identified as ``minActiveMembers`` in the REST API, so this
        is just a convenience key for users of Ansible (where the ``bigip_virtual_server``
        parameter is called ``priority_group_activation``.

        Returns:
            int: Priority number assigned to the pool members.
        """
        return self._values['minimum_active_members']

    @property
    def metadata(self):
        """Returns metadata associated with a pool

        An arbitrary amount of metadata may be associated with a pool. You typically
        see this used in situations where the user wants to annotate a resource, maybe
        in cases where an automation system is responsible for creating the resource.

        The metadata in the API is always stored as a list of dictionaries. We change
        this to be a simple dictionary before it is returned to the user.

        Returns:
            dict: A dictionary of key/value pairs where the key is the metadata name
                  and the value is the metadata value.
        """
        if self._values['metadata'] is None:
            return None
        result = dict([(k['name'], k['value']) for k in self._values['metadata']])
        return result

    @property
    def members(self):
        if not self._values['members']:
            return None
        result = []
        for member in self._values['members']:
            member['connection_limit'] = member.pop('connectionLimit', None)
            member['dynamic_ratio'] = member.pop('dynamicRatio', None)
            member['full_path'] = member.pop('fullPath', None)
            member['inherit_profile'] = member.pop('inheritProfile', None)
            member['priority_group'] = member.pop('priorityGroup', None)
            member['rate_limit'] = member.pop('rateLimit', None)
            if 'fqdn' in member and 'autopopulate' in member['fqdn']:
                if member['fqdn']['autopopulate'] == 'enabled':
                    member['fqdn_autopopulate'] = 'yes'
                elif member['fqdn']['autopopulate'] == 'disabled':
                    member['fqdn_autopopulate'] = 'no'
                del member['fqdn']
            for key in ['ephemeral', 'inherit_profile', 'logging', 'rate_limit']:
                tmp = flatten_boolean(member[key])
                member[key] = tmp
            if 'profiles' in member:
                member['encapsulation_profile'] = [x['name'] for x in member['profiles']][0]
                del member['profiles']
            if 'monitor' in member:
                monitors = member.pop('monitor')
                if monitors is not None:
                    try:
                        member['monitors'] = re.findall('/[\\w-]+/[^\\s}]+', monitors)
                    except Exception:
                        member['monitors'] = [monitors.strip()]
            session = member.pop('session')
            state = member.pop('state')
            member['real_session'] = session
            member['real_state'] = state
            if state in ['user-up', 'unchecked', 'fqdn-up-no-addr', 'fqdn-up'] and session in ['user-enabled']:
                member['state'] = 'present'
            elif state in ['user-down'] and session in ['user-disabled']:
                member['state'] = 'forced_offline'
            elif state in ['up', 'checking'] and session in ['monitor-enabled']:
                member['state'] = 'present'
            elif state in ['down'] and session in ['monitor-enabled']:
                member['state'] = 'offline'
            else:
                member['state'] = 'disabled'
            self._remove_internal_keywords(member)
            member = dict([(k, v) for k, v in iteritems(member) if v is not None])
            result.append(member)
        return result

    @property
    def monitors(self):
        if self._values['monitors'] is None:
            return None
        try:
            result = re.findall('/[\\w-]+/[^\\s}]+', self._values['monitors'])
            return result
        except Exception:
            return [self._values['monitors'].strip()]