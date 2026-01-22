from __future__ import (absolute_import, division, print_function)
from ansible.plugins.action import ActionBase
from ansible.errors import AnsibleActionFail
from ansible_collections.cisco.meraki.plugins.plugin_utils.meraki import (
from ansible_collections.cisco.meraki.plugins.plugin_utils.exceptions import (
class OrganizationsAlertsProfiles(object):

    def __init__(self, params, meraki):
        self.meraki = meraki
        self.new_object = dict(type=params.get('type'), alertCondition=params.get('alertCondition'), recipients=params.get('recipients'), networkTags=params.get('networkTags'), description=params.get('description'), organization_id=params.get('organizationId'), enabled=params.get('enabled'), alert_config_id=params.get('alertConfigId'))

    def create_params(self):
        new_object_params = {}
        if self.new_object.get('type') is not None or self.new_object.get('type') is not None:
            new_object_params['type'] = self.new_object.get('type') or self.new_object.get('type')
        if self.new_object.get('alertCondition') is not None or self.new_object.get('alert_condition') is not None:
            new_object_params['alertCondition'] = self.new_object.get('alertCondition') or self.new_object.get('alert_condition')
        if self.new_object.get('recipients') is not None or self.new_object.get('recipients') is not None:
            new_object_params['recipients'] = self.new_object.get('recipients') or self.new_object.get('recipients')
        if self.new_object.get('networkTags') is not None or self.new_object.get('network_tags') is not None:
            new_object_params['networkTags'] = self.new_object.get('networkTags') or self.new_object.get('network_tags')
        if self.new_object.get('description') is not None or self.new_object.get('description') is not None:
            new_object_params['description'] = self.new_object.get('description') or self.new_object.get('description')
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        return new_object_params

    def delete_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        if self.new_object.get('alertConfigId') is not None or self.new_object.get('alert_config_id') is not None:
            new_object_params['alertConfigId'] = self.new_object.get('alertConfigId') or self.new_object.get('alert_config_id')
        return new_object_params

    def update_by_id_params(self):
        new_object_params = {}
        if self.new_object.get('enabled') is not None or self.new_object.get('enabled') is not None:
            new_object_params['enabled'] = self.new_object.get('enabled')
        if self.new_object.get('type') is not None or self.new_object.get('type') is not None:
            new_object_params['type'] = self.new_object.get('type') or self.new_object.get('type')
        if self.new_object.get('alertCondition') is not None or self.new_object.get('alert_condition') is not None:
            new_object_params['alertCondition'] = self.new_object.get('alertCondition') or self.new_object.get('alert_condition')
        if self.new_object.get('recipients') is not None or self.new_object.get('recipients') is not None:
            new_object_params['recipients'] = self.new_object.get('recipients') or self.new_object.get('recipients')
        if self.new_object.get('networkTags') is not None or self.new_object.get('network_tags') is not None:
            new_object_params['networkTags'] = self.new_object.get('networkTags') or self.new_object.get('network_tags')
        if self.new_object.get('description') is not None or self.new_object.get('description') is not None:
            new_object_params['description'] = self.new_object.get('description') or self.new_object.get('description')
        if self.new_object.get('organizationId') is not None or self.new_object.get('organization_id') is not None:
            new_object_params['organizationId'] = self.new_object.get('organizationId') or self.new_object.get('organization_id')
        if self.new_object.get('alertConfigId') is not None or self.new_object.get('alert_config_id') is not None:
            new_object_params['alertConfigId'] = self.new_object.get('alertConfigId') or self.new_object.get('alert_config_id')
        return new_object_params

    def get_object_by_name(self, name):
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
        o_id = o_id or self.new_object.get('alert_config_id') or self.new_object.get('alertConfigId')
        name = self.new_object.get('name')
        if o_id:
            prev_obj = self.get_object_by_name(o_id)
            id_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if not id_exists and name:
            prev_obj = self.get_object_by_name(name)
            name_exists = prev_obj is not None and isinstance(prev_obj, dict)
        if name_exists:
            _id = prev_obj.get('id')
            _id = _id or prev_obj.get('alertConfigId')
            if id_exists and name_exists and (o_id != _id):
                raise InconsistentParameters("The 'id' and 'name' params don't refer to the same object")
            if _id:
                self.new_object.update(dict(id=_id))
                self.new_object.update(dict(alertConfigId=_id))
        it_exists = prev_obj is not None and isinstance(prev_obj, dict)
        return (it_exists, prev_obj)

    def requires_update(self, current_obj):
        requested_obj = self.new_object
        obj_params = [('type', 'type'), ('alertCondition', 'alertCondition'), ('recipients', 'recipients'), ('networkTags', 'networkTags'), ('description', 'description'), ('organizationId', 'organizationId'), ('enabled', 'enabled'), ('alertConfigId', 'alertConfigId')]
        return any((not meraki_compare_equality(current_obj.get(meraki_param), requested_obj.get(ansible_param)) for meraki_param, ansible_param in obj_params))

    def create(self):
        result = self.meraki.exec_meraki(family='organizations', function='createOrganizationAlertsProfile', params=self.create_params(), op_modifies=True)
        return result

    def update(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('alertConfigId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('alertConfigId')
            if id_:
                self.new_object.update(dict(alertconfigid=id_))
        result = self.meraki.exec_meraki(family='organizations', function='updateOrganizationAlertsProfile', params=self.update_by_id_params(), op_modifies=True)
        return result

    def delete(self):
        id = self.new_object.get('id')
        id = id or self.new_object.get('alertConfigId')
        name = self.new_object.get('name')
        result = None
        if not id:
            prev_obj_name = self.get_object_by_name(name)
            id_ = None
            if prev_obj_name:
                id_ = prev_obj_name.get('id')
                id_ = id_ or prev_obj_name.get('alertConfigId')
            if id_:
                self.new_object.update(dict(alertconfigid=id_))
        result = self.meraki.exec_meraki(family='organizations', function='deleteOrganizationAlertsProfile', params=self.delete_by_id_params())
        return result