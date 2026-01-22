from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.dnac.plugins.plugin_utils.dnac import (
from ansible_collections.cisco.dnac.plugins.plugin_utils.exceptions import (
class TagMember(object):

    def __init__(self, params, dnac):
        self.dnac = dnac
        self.new_object = dict(payload=params.get('payload'), object=params.get('object'), id=params.get('id'), member_id=params.get('memberId'), member_type=params.get('memberType'))

    def create_params(self):
        new_object_params = {}
        new_object_params['payload'] = self.new_object.get('payload')
        new_object_params['id'] = self.new_object.get('id')
        new_object_params['object'] = self.new_object.get('object')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        new_object_params['id'] = self.new_object.get('id')
        new_object_params['member_id'] = self.new_object.get('member_id')
        return new_object_params

    def get_object_by_name(self, name):
        result = None
        return result

    def get_object_by_id(self, id):
        result = None
        try:
            items = self.dnac.exec(family='tag', function='get_tag_members_by_id', params={'id': id, 'memberType': self.new_object.get('member_type')})
            if isinstance(items, dict):
                if 'response' in items:
                    items = items.get('response')
            result = get_dict_result(items, 'id', id)
        except Exception:
            result = None
        return result

    def exists(self):
        prev_obj = None
        id_exists = False
        name_exists = False
        o_id = self.new_object.get('id')
        o_id = o_id or self.new_object.get('member_id')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_id(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('memberId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(member_id=_id))
            if _id:
                prev_obj = self.get_object_by_id(_id)
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('object', 'object'), ('id', 'id'), ('memberId', 'member_id')]
        return any((not dnac_compare_equality(current_obj.get(dnac_param), requested_obj.get(ansible_param)) for dnac_param, ansible_param in obj_params))

    def create(self):
        result = self.dnac.exec(family='tag', function='add_members_to_the_tag', params=self.create_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('member_id')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('memberId')
            if id_:
                self.new_object.update(dict(member_id=id_))
        result = self.dnac.exec(family='tag', function='remove_tag_member', params=self.delete_by_id_params())
        return result