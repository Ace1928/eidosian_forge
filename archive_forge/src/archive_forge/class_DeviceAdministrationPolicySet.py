from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
from ansible_collections.cisco.ise.plugins.plugin_utils.exceptions import (
class DeviceAdministrationPolicySet(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(condition=params.get('condition'), default=params.get('default'), description=params.get('description'), hit_counts=params.get('hitCounts'), id=params.get('id'), is_proxy=params.get('isProxy'), link=params.get('link'), name=params.get('name'), rank=params.get('rank'), service_name=params.get('serviceName'), state=params.get('state_'))

    def get_object_by_name(self, name):
        result = None
        items = self.ise.exec(family='device_administration_policy_set', function='get_device_admin_policy_sets').response.get('response', []) or []
        result = get_dict_result(items, 'name', name)
        return result

    def get_object_by_id(self, id):
        try:
            result = self.ise.exec(family='device_administration_policy_set', function='get_device_admin_policy_set_by_id', params={'id': id}, handle_func_exception=False).response['response']
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
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
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('condition', 'condition'), ('default', 'default'), ('description', 'description'), ('hitCounts', 'hit_counts'), ('id', 'id'), ('isProxy', 'is_proxy'), ('link', 'link'), ('name', 'name'), ('rank', 'rank'), ('serviceName', 'service_name'), ('state_', 'state')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        result = self.ise.exec(family='device_administration_policy_set', function='create_device_admin_policy_set', params=self.new_object).response
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='device_administration_policy_set', function='update_device_admin_policy_set_by_id', params=self.new_object).response
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='device_administration_policy_set', function='delete_device_admin_policy_set_by_id', params=self.new_object).response
        return result