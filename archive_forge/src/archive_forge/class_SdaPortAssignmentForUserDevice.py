from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
class SdaPortAssignmentForUserDevice(object):

    def __init__(self, params, dnac):
        self.dnac = dnac
        self.new_object = dict(siteNameHierarchy=params.get('siteNameHierarchy'), deviceManagementIpAddress=params.get('deviceManagementIpAddress'), interfaceName=params.get('interfaceName'), interfaceNames=params.get('interfaceNames'), dataIpAddressPoolName=params.get('dataIpAddressPoolName'), voiceIpAddressPoolName=params.get('voiceIpAddressPoolName'), authenticateTemplateName=params.get('authenticateTemplateName'), scalableGroupName=params.get('scalableGroupName'), interfaceDescription=params.get('interfaceDescription'), device_management_ip_address=params.get('deviceManagementIpAddress'), interface_name=params.get('interfaceName'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        new_object_params['device_management_ip_address'] = self.new_object.get('deviceManagementIpAddress') or self.new_object.get('device_management_ip_address')
        new_object_params['interface_name'] = self.new_object.get('interfaceName') or self.new_object.get('interface_name')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        new_object_params['siteNameHierarchy'] = self.new_object.get('siteNameHierarchy')
        new_object_params['deviceManagementIpAddress'] = self.new_object.get('deviceManagementIpAddress')
        new_object_params['interfaceName'] = self.new_object.get('interfaceName')
        new_object_params['interfaceNames'] = self.new_object.get('interfaceNames')
        new_object_params['dataIpAddressPoolName'] = self.new_object.get('dataIpAddressPoolName')
        new_object_params['voiceIpAddressPoolName'] = self.new_object.get('voiceIpAddressPoolName')
        new_object_params['authenticateTemplateName'] = self.new_object.get('authenticateTemplateName')
        new_object_params['scalableGroupName'] = self.new_object.get('scalableGroupName')
        new_object_params['interfaceDescription'] = self.new_object.get('interfaceDescription')
        return new_object_params

    def delete_all_params(self):
        new_object_params = {}
        new_object_params['device_management_ip_address'] = self.new_object.get('device_management_ip_address')
        new_object_params['interface_name'] = self.new_object.get('interface_name')
        return new_object_params

    def get_object_by_name(self, name, is_absent=False):
        result = None
        try:
            items = self.dnac.exec(family='sda', function='get_port_assignment_for_user_device', params=self.get_all_params(name=name))
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
                if isinstance(items, dict) and items.get('status') == 'failed':
                    if is_absent:
                        raise AnsibleSDAException(response=items)
                    result = None
                    return result
            result = get_dict_result(items, 'name', name)
        except Exception:
            if is_absent:
                raise
            result = None
        return result

    def get_object_by_id(self, id):
        result = None
        return result

    def exists(self, is_absent=False):
        name = self.new_object.get('name')
        prev_obj = self.get_object_by_name(name, is_absent=is_absent)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict) and (prev_obj.get('status') != 'failed')
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('siteNameHierarchy', 'siteNameHierarchy'), ('deviceManagementIpAddress', 'deviceManagementIpAddress'), ('interfaceName', 'interfaceName'), ('interfaceNames', 'interfaceNames'), ('dataIpAddressPoolName', 'dataIpAddressPoolName'), ('voiceIpAddressPoolName', 'voiceIpAddressPoolName'), ('authenticateTemplateName', 'authenticateTemplateName'), ('scalableGroupName', 'scalableGroupName'), ('interfaceDescription', 'interfaceDescription'), ('deviceManagementIpAddress', 'device_management_ip_address'), ('interfaceName', 'interface_name')]
        return any((not dnac_compare_equality(current_obj.get(dnac_param), requested_obj.get(ansible_param)) for dnac_param, ansible_param in obj_params))

    def create(self):
        result = self.dnac.exec(family='sda', function='add_port_assignment_for_user_device', params=self.create_params(), op_modifies=True)
        if isinstance(result, dict):
            if 'response' in result:
                result = result.get('response')
            if isinstance(result, dict) and result.get('status') == 'failed':
                raise AnsibleSDAException(response=result)
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        result = self.dnac.exec(family='sda', function='delete_port_assignment_for_user_device', params=self.delete_all_params())
        return result