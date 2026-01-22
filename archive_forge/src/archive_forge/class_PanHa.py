from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
from ansible_collections.cisco.ise.plugins.plugin_utils.exceptions import (
class PanHa(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(is_enabled=params.get('isEnabled'), primary_health_check_node=params.get('primaryHealthCheckNode'), secondary_health_check_node=params.get('secondaryHealthCheckNode'), polling_interval=params.get('pollingInterval'), failed_attempts=params.get('failedAttempts'))

    def get_object_by_name(self, name):
        result = None
        items = self.ise.exec(family='pan_ha', function='get_pan_ha_status').response['response']
        result = get_dict_result(items, 'name', name)
        return result

    def get_object_by_id(self, id):
        result = None
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
        o_id = self.new_object.get('id')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('isEnabled', 'is_enabled'), ('primaryHealthCheckNode', 'primary_health_check_node'), ('secondaryHealthCheckNode', 'secondary_health_check_node'), ('pollingInterval', 'polling_interval'), ('failedAttempts', 'failed_attempts')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        result = self.ise.exec(family='pan_ha', function='enable_pan_ha', params=self.new_object).response
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        result = self.ise.exec(family='pan_ha', function='disable_pan_ha', params=self.new_object).response
        return result