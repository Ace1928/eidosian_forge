from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
class InternalUser(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(name=params.get('name'), description=params.get('description'), enabled=params.get('enabled'), email=params.get('email'), password=params.get('password'), first_name=params.get('firstName'), last_name=params.get('lastName'), change_password=params.get('changePassword'), identity_groups=params.get('identityGroups'), expiry_date_enabled=params.get('expiryDateEnabled'), expiry_date=params.get('expiryDate'), enable_password=params.get('enablePassword'), custom_attributes=params.get('customAttributes'), password_idstore=params.get('passwordIDStore'), id=params.get('id'))

    def get_object_by_name(self, name):
        try:
            result = self.ise.exec(family='internal_user', function='get_internal_user_by_name', params={'name': name}, handle_func_exception=False).response['InternalUser']
            result = get_dict_result(result, 'name', name)
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
        return result

    def get_object_by_id(self, id):
        try:
            result = self.ise.exec(family='internal_user', function='get_internal_user_by_id', params={'id': id}, handle_func_exception=False).response['InternalUser']
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
        return result

    def exists(self):
        result = False
        prev_obj = None
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        if id:
            prev_obj = self.get_object_by_id(id)
            result = prev_obj is not None and isinstance(prev_obj, dict)
        elif name:
            prev_obj = self.get_object_by_name(name)
            result = prev_obj is not None and isinstance(prev_obj, dict)
        return (result, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        force_change = False
        change_params = [('change_password', bool)]
        for change_param, type_ in change_params:
            requested_obj_value = requested_obj.get(change_param)
            if isinstance(requested_obj_value, type_):
                if requested_obj_value:
                    force_change = True
                    break
                else:
                    pass
            else:
                pass
        if force_change:
            return force_change
        obj_params = [('name', 'name'), ('description', 'description'), ('enabled', 'enabled'), ('email', 'email'), ('password', 'password'), ('firstName', 'first_name'), ('lastName', 'last_name'), ('changePassword', 'change_password'), ('identityGroups', 'identity_groups'), ('expiryDateEnabled', 'expiry_date_enabled'), ('expiryDate', 'expiry_date'), ('enablePassword', 'enable_password'), ('customAttributes', 'custom_attributes'), ('passwordIDStore', 'password_idstore'), ('id', 'id')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        result = self.ise.exec(family='internal_user', function='create_internal_user', params=self.new_object).response
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        change_password = self.new_object.get('change_password')
        result = None
        if id:
            try:
                result = self.ise.exec(family='internal_user', function='update_internal_user_by_id', params=self.new_object, handle_func_exception=False).response
            except exceptions.ApiError as e:
                if not change_password and "Password can't be set to one of the earlier" in e.message:
                    self.ise.object_modify_result(changed=False, result='Object already present, update was attempted but failed because of password')
                    result = {'_changed_': True}
                elif not change_password and "Password can't be set to one of the earlier" in e.details_str:
                    self.ise.object_modify_result(changed=False, result='Object already present, update was attempted but failed because of password')
                    result = {'_changed_': True}
                else:
                    raise e
        elif name:
            try:
                result = self.ise.exec(family='internal_user', function='update_internal_user_by_name', params=self.new_object, handle_func_exception=False).response
            except exceptions.ApiError as e:
                if not change_password and "Password can't be set to one of the earlier" in e.message:
                    self.ise.object_modify_result(changed=False, result='Object already present, update was attempted but failed because of password')
                    result = {'_changed_': True}
                elif not change_password and "Password can't be set to one of the earlier" in e.details_str:
                    self.ise.object_modify_result(changed=False, result='Object already present, update was attempted but failed because of password')
                    result = {'_changed_': True}
                else:
                    raise e
        return result

    def delete(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if id:
            result = self.ise.exec(family='internal_user', function='delete_internal_user_by_id', params=self.new_object).response
        elif name:
            result = self.ise.exec(family='internal_user', function='delete_internal_user_by_name', params=self.new_object).response
        return result