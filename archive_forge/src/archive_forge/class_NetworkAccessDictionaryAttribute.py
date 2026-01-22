from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
from ansible_collections.cisco.ise.plugins.plugin_utils.exceptions import (
class NetworkAccessDictionaryAttribute(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(allowed_values=params.get('allowedValues'), data_type=params.get('dataType'), description=params.get('description'), dictionary_name=params.get('dictionaryName'), direction_type=params.get('directionType'), id=params.get('id'), internal_name=params.get('internalName'), name=params.get('name'))

    def get_object_by_name(self, name, dictionary_name):
        try:
            result = self.ise.exec(family='network_access_dictionary_attribute', function='get_network_access_dictionary_attribute_by_name', params={'name': name, 'dictionary_name': dictionary_name}, handle_func_exception=False).response['response']
            result = get_dict_result(result, 'name', name)
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
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
        dictionary_name = self.new_object.get('dictionary_name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name, dictionary_name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('allowedValues', 'allowed_values'), ('dataType', 'data_type'), ('description', 'description'), ('dictionaryName', 'dictionary_name'), ('directionType', 'direction_type'), ('id', 'id'), ('internalName', 'internal_name'), ('name', 'name')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        result = self.ise.exec(family='network_access_dictionary_attribute', function='create_network_access_dictionary_attribute', params=self.new_object).response
        return result

    def update(self):
        result = self.ise.exec(family='network_access_dictionary_attribute', function='update_network_access_dictionary_attribute_by_name', params=self.new_object).response
        return result

    def delete(self):
        result = self.ise.exec(family='network_access_dictionary_attribute', function='delete_network_access_dictionary_attribute_by_name', params=self.new_object).response
        return result