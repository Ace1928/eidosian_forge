from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
class ReserveIpSubpool(object):

    def __init__(self, params, dnac):
        self.dnac = dnac
        self.new_object = dict(name=params.get('name'), type=params.get('type'), ipv6AddressSpace=params.get('ipv6AddressSpace'), ipv4GlobalPool=params.get('ipv4GlobalPool'), ipv4Prefix=params.get('ipv4Prefix'), ipv4PrefixLength=params.get('ipv4PrefixLength'), ipv4Subnet=params.get('ipv4Subnet'), ipv4GateWay=params.get('ipv4GateWay'), ipv4DhcpServers=params.get('ipv4DhcpServers'), ipv4DnsServers=params.get('ipv4DnsServers'), ipv6GlobalPool=params.get('ipv6GlobalPool'), ipv6Prefix=params.get('ipv6Prefix'), ipv6PrefixLength=params.get('ipv6PrefixLength'), ipv6Subnet=params.get('ipv6Subnet'), ipv6GateWay=params.get('ipv6GateWay'), ipv6DhcpServers=params.get('ipv6DhcpServers'), ipv6DnsServers=params.get('ipv6DnsServers'), ipv4TotalHost=params.get('ipv4TotalHost'), ipv6TotalHost=params.get('ipv6TotalHost'), slaacSupport=params.get('slaacSupport'), site_id=params.get('siteId'), id=params.get('id'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        new_object_params['site_id'] = self.new_object.get('siteId') or self.new_object.get('site_id')
        new_object_params['offset'] = self.new_object.get('offset')
        new_object_params['limit'] = self.new_object.get('limit')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        new_object_params['name'] = self.new_object.get('name')
        new_object_params['type'] = self.new_object.get('type')
        new_object_params['ipv6AddressSpace'] = self.new_object.get('ipv6AddressSpace')
        new_object_params['ipv4GlobalPool'] = self.new_object.get('ipv4GlobalPool')
        new_object_params['ipv4Prefix'] = self.new_object.get('ipv4Prefix')
        new_object_params['ipv4PrefixLength'] = self.new_object.get('ipv4PrefixLength')
        new_object_params['ipv4Subnet'] = self.new_object.get('ipv4Subnet')
        new_object_params['ipv4GateWay'] = self.new_object.get('ipv4GateWay')
        new_object_params['ipv4DhcpServers'] = self.new_object.get('ipv4DhcpServers')
        new_object_params['ipv4DnsServers'] = self.new_object.get('ipv4DnsServers')
        new_object_params['ipv6GlobalPool'] = self.new_object.get('ipv6GlobalPool')
        new_object_params['ipv6Prefix'] = self.new_object.get('ipv6Prefix')
        new_object_params['ipv6PrefixLength'] = self.new_object.get('ipv6PrefixLength')
        new_object_params['ipv6Subnet'] = self.new_object.get('ipv6Subnet')
        new_object_params['ipv6GateWay'] = self.new_object.get('ipv6GateWay')
        new_object_params['ipv6DhcpServers'] = self.new_object.get('ipv6DhcpServers')
        new_object_params['ipv6DnsServers'] = self.new_object.get('ipv6DnsServers')
        new_object_params['ipv4TotalHost'] = self.new_object.get('ipv4TotalHost')
        new_object_params['ipv6TotalHost'] = self.new_object.get('ipv6TotalHost')
        new_object_params['slaacSupport'] = self.new_object.get('slaacSupport')
        new_object_params['site_id'] = self.new_object.get('site_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        new_object_params['id'] = self.new_object.get('id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        new_object_params['name'] = self.new_object.get('name')
        new_object_params['ipv6AddressSpace'] = self.new_object.get('ipv6AddressSpace')
        new_object_params['ipv4DhcpServers'] = self.new_object.get('ipv4DhcpServers')
        new_object_params['ipv4DnsServers'] = self.new_object.get('ipv4DnsServers')
        new_object_params['ipv6GlobalPool'] = self.new_object.get('ipv6GlobalPool')
        new_object_params['ipv6Prefix'] = self.new_object.get('ipv6Prefix')
        new_object_params['ipv6PrefixLength'] = self.new_object.get('ipv6PrefixLength')
        new_object_params['ipv6Subnet'] = self.new_object.get('ipv6Subnet')
        new_object_params['ipv6GateWay'] = self.new_object.get('ipv6GateWay')
        new_object_params['ipv6DhcpServers'] = self.new_object.get('ipv6DhcpServers')
        new_object_params['ipv6DnsServers'] = self.new_object.get('ipv6DnsServers')
        new_object_params['ipv6TotalHost'] = self.new_object.get('ipv6TotalHost')
        new_object_params['slaacSupport'] = self.new_object.get('slaacSupport')
        new_object_params['ipv4GateWay'] = self.new_object.get('ipv4GateWay')
        new_object_params['id'] = self.new_object.get('id')
        new_object_params['site_id'] = self.new_object.get('site_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.dnac.exec(family='network_settings', function='get_reserve_ip_subpool', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'name', name)
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        try:
            items = self.dnac.exec(family='network_settings', function='get_reserve_ip_subpool', params=self.get_all_params(id=id))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'id', id)
        except Exception:
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('site_id')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('siteId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(site_id=_id))
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('name', 'name'), ('type', 'type'), ('ipv6AddressSpace', 'ipv6AddressSpace'), ('ipv4GlobalPool', 'ipv4GlobalPool'), ('ipv4Prefix', 'ipv4Prefix'), ('ipv4PrefixLength', 'ipv4PrefixLength'), ('ipv4Subnet', 'ipv4Subnet'), ('ipv4GateWay', 'ipv4GateWay'), ('ipv4DhcpServers', 'ipv4DhcpServers'), ('ipv4DnsServers', 'ipv4DnsServers'), ('ipv6GlobalPool', 'ipv6GlobalPool'), ('ipv6Prefix', 'ipv6Prefix'), ('ipv6PrefixLength', 'ipv6PrefixLength'), ('ipv6Subnet', 'ipv6Subnet'), ('ipv6GateWay', 'ipv6GateWay'), ('ipv6DhcpServers', 'ipv6DhcpServers'), ('ipv6DnsServers', 'ipv6DnsServers'), ('ipv4TotalHost', 'ipv4TotalHost'), ('ipv6TotalHost', 'ipv6TotalHost'), ('slaacSupport', 'slaacSupport'), ('siteId', 'site_id'), ('id', 'id')]
        return any((not dnac_compare_equality(current_obj.get(dnac_param), requested_obj.get(ansible_param)) for dnac_param, ansible_param in obj_params))

    def create(self):
        result = self.dnac.exec(family='network_settings', function='reserve_ip_subpool', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('site_id')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('siteId')
            if id_:
                self.new_object.update(dict(site_id=id_))
        result = self.dnac.exec(family='network_settings', function='update_reserve_ip_subpool', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('site_id')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('siteId')
            if id_:
                self.new_object.update(dict(id=id_))
        result = self.dnac.exec(family='network_settings', function='release_reserve_ip_subpool', params=self.delete_by_id_params())
        return result