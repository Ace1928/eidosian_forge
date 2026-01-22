from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class NetworksMerakiAuthUsers(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(email=params.get('email'), name=params.get('name'), password=params.get('password'), accountType=params.get('accountType'), emailPasswordToUser=params.get('emailPasswordToUser'), isAdmin=params.get('isAdmin'), authorizations=params.get('authorizations'), networkId=params.get('networkId'), merakiAuthUserId=params.get('merakiAuthUserId'))

    def get_all_params(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def get_params_by_id(self, name=None, id=None):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('merakiAuthUserId') is not None or self.new_object.get('meraki_auth_user_id') is not None:
            new_object_params['merakiAuthUserId'] = self.new_object.get('merakiAuthUserId') or self.new_object.get('meraki_auth_user_id')
        return new_object_params

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('email') is not None or self.new_object.get('email') is not None:
            new_object_params['email'] = self.new_object.get('email') or self.new_object.get('email')
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('password') is not None or self.new_object.get('password') is not None:
            new_object_params['password'] = self.new_object.get('password') or self.new_object.get('password')
        if self.new_object.get('accountType') is not None or self.new_object.get('account_type') is not None:
            new_object_params['accountType'] = self.new_object.get('accountType') or self.new_object.get('account_type')
        if self.new_object.get('emailPasswordToUser') is not None or self.new_object.get('email_password_to_user') is not None:
            new_object_params['emailPasswordToUser'] = self.new_object.get('emailPasswordToUser')
        if self.new_object.get('isAdmin') is not None or self.new_object.get('is_admin') is not None:
            new_object_params['isAdmin'] = self.new_object.get('isAdmin')
        if self.new_object.get('authorizations') is not None or self.new_object.get('authorizations') is not None:
            new_object_params['authorizations'] = self.new_object.get('authorizations') or self.new_object.get('authorizations')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('merakiAuthUserId') is not None or self.new_object.get('meraki_auth_user_id') is not None:
            new_object_params['merakiAuthUserId'] = self.new_object.get('merakiAuthUserId') or self.new_object.get('meraki_auth_user_id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('name') is not None or self.new_object.get('name') is not None:
            new_object_params['name'] = self.new_object.get('name') or self.new_object.get('name')
        if self.new_object.get('password') is not None or self.new_object.get('password') is not None:
            new_object_params['password'] = self.new_object.get('password') or self.new_object.get('password')
        if self.new_object.get('emailPasswordToUser') is not None or self.new_object.get('email_password_to_user') is not None:
            new_object_params['emailPasswordToUser'] = self.new_object.get('emailPasswordToUser')
        if self.new_object.get('authorizations') is not None or self.new_object.get('authorizations') is not None:
            new_object_params['authorizations'] = self.new_object.get('authorizations') or self.new_object.get('authorizations')
        if self.new_object.get('networkId') is not None or self.new_object.get('network_id') is not None:
            new_object_params['networkId'] = self.new_object.get('networkId') or self.new_object.get('network_id')
        if self.new_object.get('merakiAuthUserId') is not None or self.new_object.get('meraki_auth_user_id') is not None:
            new_object_params['merakiAuthUserId'] = self.new_object.get('merakiAuthUserId') or self.new_object.get('meraki_auth_user_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        try:
            items = self.meraki.exec_meraki(family='networks', function='getNetworkMerakiAuthUsers', params=self.get_all_params(name=name))
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
            items = self.meraki.exec_meraki(family='networks', function='getNetworkMerakiAuthUser', params=self.get_params_by_id())
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'merakiAuthUserId', id)
        except Exception as e:
            print('Error: ', e)
            result = None
        return result

    def exists(self):
        id_exists = False
        name_exists = False
        prev_obj = None
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('meraki_auth_user_id') or self.new_object.get('merakiAuthUserId')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('merakiAuthUserId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(merakiAuthUserId=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('email', 'email'), ('name', 'name'), ('accountType', 'accountType'), ('emailPasswordToUser', 'emailPasswordToUser'), ('isAdmin', 'isAdmin'), ('authorizations', 'authorizations'), ('networkId', 'networkId'), ('merakiAuthUserId', 'merakiAuthUserId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='networks', function='createNetworkMerakiAuthUser', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('merakiAuthUserId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('merakiAuthUserId')
            if id_:
                self.new_object.update(dict(merakiAuthUserId=id_))
        result = self.meraki.exec_meraki(family='networks', function='updateNetworkMerakiAuthUser', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('merakiAuthUserId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('merakiAuthUserId')
            if id_:
                self.new_object.update(dict(merakiAuthUserId=id_))
        result = self.meraki.exec_meraki(family='networks', function='deleteNetworkMerakiAuthUser', params=self.delete_by_id_params())
        return result