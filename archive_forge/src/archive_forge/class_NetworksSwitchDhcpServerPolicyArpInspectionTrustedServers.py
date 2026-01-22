from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksSwitchDhcpServerPolicyArpInspectionTrustedServers(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(mac=params.get('mac'), vlan=params.get('vlan'), ipv4=params.get('ipv4'), networkId=params.get('networkId'), trustedServerId=params.get('trustedServerId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('perPage') is not None or self.new_object.get('per_page') is not None:
            new_object_params['perPage'] = self.new_object.get('perPage') or self.new_object.get('per_page')
        new_object_params['total_pages'] = -1
        if self.new_object.get('startingAfter') is not None or self.new_object.get('starting_after') is not None:
            new_object_params['startingAfter'] = self.new_object.get('startingAfter') or self.new_object.get('starting_after')
        if self.new_object.get('endingBefore') is not None or self.new_object.get('ending_before') is not None:
            new_object_params['endingBefore'] = self.new_object.get('endingBefore') or self.new_object.get('ending_before')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('mac') is not None or self.new_object.get('mac') is not None:
            new_object_params['mac'] = self.new_object.get('mac') or self.new_object.get('mac')
        if self.new_object.get('vlan') is not None or self.new_object.get('vlan') is not None:
            new_object_params['vlan'] = self.new_object.get('vlan') or self.new_object.get('vlan')
        if self.new_object.get('ipv4') is not None or self.new_object.get('ipv4') is not None:
            new_object_params['ipv4'] = self.new_object.get('ipv4') or self.new_object.get('ipv4')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('trustedServerId') is not None or self.new_object.get('trusted_server_id') is not None:
            new_object_params['trustedServerId'] = self.new_object.get('trustedServerId') or self.new_object.get('trusted_server_id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('mac') is not None or self.new_object.get('mac') is not None:
            new_object_params['mac'] = self.new_object.get('mac') or self.new_object.get('mac')
        if self.new_object.get('vlan') is not None or self.new_object.get('vlan') is not None:
            new_object_params['vlan'] = self.new_object.get('vlan') or self.new_object.get('vlan')
        if self.new_object.get('ipv4') is not None or self.new_object.get('ipv4') is not None:
            new_object_params['ipv4'] = self.new_object.get('ipv4') or self.new_object.get('ipv4')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('trustedServerId') is not None or self.new_object.get('trusted_server_id') is not None:
            new_object_params['trustedServerId'] = self.new_object.get('trustedServerId') or self.new_object.get('trusted_server_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='switch', function='getNetworkSwitchDhcpServerPolicyArpInspectionTrustedServers', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'name', name)
            if result is None:
                result = items
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        try:
            items = self.meraki.exec_meraki(family='switch', function='getNetworkSwitchDhcpServerPolicyArpInspectionTrustedServers', params=self.get_all_params(id=id))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'id', id)
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('trusted_server_id') or self.new_object.get('trustedServerId')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('trustedServerId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(trustedServerId=_id))
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('mac', 'mac'), ('vlan', 'vlan'), ('ipv4', 'ipv4'), ('networkId', 'networkId'), ('trustedServerId', 'trustedServerId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='switch', function='createNetworkSwitchDhcpServerPolicyArpInspectionTrustedServer', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('trustedServerId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('trustedServerId')
            if id_:
                self.new_object.update(dict(trustedServerId=id_))
        result = self.meraki.exec_meraki(family='switch', function='updateNetworkSwitchDhcpServerPolicyArpInspectionTrustedServer', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('trustedServerId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('trustedServerId')
            if id_:
                self.new_object.update(dict(trustedServerId=id_))
        result = self.meraki.exec_meraki(family='switch', function='deleteNetworkSwitchDhcpServerPolicyArpInspectionTrustedServer', params=self.delete_by_id_params())
        return result