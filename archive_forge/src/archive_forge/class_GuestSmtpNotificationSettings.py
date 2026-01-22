from __future__ import absolute_import, division, print_function
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.ise.plugins.plugin_utils.ise import (
from ansible_collections.cisco.ise.plugins.plugin_utils.exceptions import (
class GuestSmtpNotificationSettings(object):

    def __init__(self, params, ise):
        self.ise = ise
        self.new_object = dict(smtp_server=params.get('smtpServer'), notification_enabled=params.get('notificationEnabled'), use_default_from_address=params.get('useDefaultFromAddress'), default_from_address=params.get('defaultFromAddress'), smtp_port=params.get('smtpPort'), connection_timeout=params.get('connectionTimeout'), use_tlsor_ssl_encryption=params.get('useTLSorSSLEncryption'), use_password_authentication=params.get('usePasswordAuthentication'), user_name=params.get('userName'), password=params.get('password'), id=params.get('id'))

    def get_object_by_name(self, name):
        result = None
        gen_items_responses = self.ise.exec(family='guest_smtp_notification_configuration', function='get_guest_smtp_notification_settings_generator')
        try:
            for items_response in gen_items_responses:
                items = items_response.response['SearchResult']['resources']
                result = get_dict_result(items, 'name', name)
                if result:
                    return result
        except (TypeError, AttributeError) as e:
            self.ise.fail_json(msg='An error occured when executing operation. Check the configuration of your API Settings and API Gateway settings on your ISE server. This collection assumes that the API Gateway, the ERS APIs and OpenAPIs are enabled. You may want to enable the (ise_debug: True) argument. The error was: {error}'.format(error=e))
        except Exception:
            result = None
            return result
        return result

    def get_object_by_id(self, id):
        try:
            result = self.ise.exec(family='guest_smtp_notification_configuration', function='get_guest_smtp_notification_settings_by_id', params={'id': id}, handle_func_exception=False).response['ERSGuestSmtpNotificationSettings']
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
        obj_params = [('smtpServer', 'smtp_server'), ('notificationEnabled', 'notification_enabled'), ('useDefaultFromAddress', 'use_default_from_address'), ('defaultFromAddress', 'default_from_address'), ('smtpPort', 'smtp_port'), ('connectionTimeout', 'connection_timeout'), ('useTLSorSSLEncryption', 'use_tlsor_ssl_encryption'), ('usePasswordAuthentication', 'use_password_authentication'), ('userName', 'user_name'), ('password', 'password'), ('id', 'id')]
        return any((not ise_compare_equality(current_obj.get(ise_param), requested_obj.get(ansible_param)) for ise_param, ansible_param in obj_params))

    def create(self):
        result = self.ise.exec(family='guest_smtp_notification_configuration', function='create_guest_smtp_notification_settings', params=self.new_object).response
        return result

    def update(self):
        id = self.new_object.get('id')
        name = self.new_object.get('name')
        result = None
        if not id:
            id_ = self.get_object_by_name(name).get('id')
            self.new_object.update(dict(id=id_))
        result = self.ise.exec(family='guest_smtp_notification_configuration', function='update_guest_smtp_notification_settings_by_id', params=self.new_object).response
        return result