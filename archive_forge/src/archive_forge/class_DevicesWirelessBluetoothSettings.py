from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class DevicesWirelessBluetoothSettings(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(uuid=params.get('uuid'), major=params.get('major'), minor=params.get('minor'), serial=params.get('serial'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('serial') is not None or self.new_object.get('serial') is not None:
            new_object_params['serial'] = self.new_object.get('serial')
        return new_object_params

    def update_all_params(self):
        new_object_params = {}
        if self.new_object.get('uuid') is not None or self.new_object.get('uuid') is not None:
            new_object_params['uuid'] = self.new_object.get('uuid') or self.new_object.get('uuid')
        if self.new_object.get('major') is not None or self.new_object.get('major') is not None:
            new_object_params['major'] = self.new_object.get('major') or self.new_object.get('major')
        if self.new_object.get('minor') is not None or self.new_object.get('minor') is not None:
            new_object_params['minor'] = self.new_object.get('minor') or self.new_object.get('minor')
        if self.new_object.get('serial') is not None or self.new_object.get('serial') is not None:
            new_object_params['serial'] = self.new_object.get('serial') or self.new_object.get('serial')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='wireless', function='getDeviceWirelessBluetoothSettings', params=self.get_all_params(name=name))
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
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
        o_id = self.new_object.get('serial')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_name(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('uuid', 'uuid'), ('major', 'major'), ('minor', 'minor'), ('serial', 'serial')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        result = self.meraki.exec_meraki(family='wireless', function='updateDeviceWirelessBluetoothSettings', params=self.update_all_params(), op_modifies=True)
        return result