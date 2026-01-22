from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _snake_to_camel, _camel_to_snake
class AzureRMLogAnalyticsWorkspace(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str', required=True), state=dict(type='str', default='present', choices=['present', 'absent']), location=dict(type='str'), sku=dict(type='str', default='per_gb2018', choices=['free', 'standard', 'premium', 'unlimited', 'per_node', 'per_gb2018', 'standalone']), retention_in_days=dict(type='int'), intelligence_packs=dict(type='dict'), force=dict(type='bool', default=False))
        self.results = dict(changed=False, id=None)
        self.resource_group = None
        self.name = None
        self.state = None
        self.location = None
        self.sku = None
        self.retention_in_days = None
        self.intelligence_packs = None
        self.force = None
        super(AzureRMLogAnalyticsWorkspace, self).__init__(self.module_arg_spec, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()) + ['tags']:
            setattr(self, key, kwargs[key])
        self.results = dict()
        changed = False
        if not self.location:
            resource_group = self.get_resource_group(self.resource_group)
            self.location = resource_group.location
        if self.sku == 'per_gb2018':
            self.sku = 'PerGB2018'
        else:
            self.sku = _snake_to_camel(self.sku)
        workspace = self.get_workspace()
        if not workspace and self.state == 'present':
            changed = True
            workspace = self.log_analytics_models.Workspace(sku=self.log_analytics_models.WorkspaceSku(name=self.sku), retention_in_days=self.retention_in_days, location=self.location, tags=self.tags)
            if not self.check_mode:
                workspace = self.create_workspace(workspace)
        elif workspace and self.state == 'present':
            if workspace.retention_in_days != self.retention_in_days:
                changed = True
            results = dict()
            update_tags, results['tags'] = self.update_tags(workspace.tags)
            if update_tags:
                changed = True
            if not self.check_mode and changed:
                workspace = self.log_analytics_models.Workspace(sku=self.log_analytics_models.WorkspaceSku(name=self.sku), retention_in_days=self.retention_in_days, location=self.location, tags=results['tags'])
                workspace = self.create_workspace(workspace)
        elif workspace and self.state == 'absent':
            changed = True
            workspace = None
            if not self.check_mode:
                self.delete_workspace()
        if workspace and workspace.id:
            self.results = self.to_dict(workspace)
            self.results['intelligence_packs'] = self.list_intelligence_packs()
            self.results['management_groups'] = self.list_management_groups()
            self.results['usages'] = self.list_usages()
            self.results['shared_keys'] = self.get_shared_keys()
        if workspace and workspace.id and self.intelligence_packs:
            intelligence_packs = self.results['intelligence_packs']
            for key in self.intelligence_packs.keys():
                enabled = self.intelligence_packs[key]
                for x in intelligence_packs:
                    if x['name'].lower() == key.lower():
                        if x['enabled'] != enabled:
                            changed = True
                            if not self.check_mode:
                                self.change_intelligence(x['name'], enabled)
                                x['enabled'] = enabled
                        break
        self.results['changed'] = changed
        return self.results

    def create_workspace(self, workspace):
        try:
            poller = self.log_analytics_client.workspaces.begin_create_or_update(self.resource_group, self.name, workspace)
            return self.get_poller_result(poller)
        except Exception as exc:
            self.fail('Error when creating workspace {0} - {1}'.format(self.name, exc.message or str(exc)))

    def get_workspace(self):
        try:
            return self.log_analytics_client.workspaces.get(self.resource_group, self.name)
        except ResourceNotFoundError:
            pass

    def delete_workspace(self):
        try:
            self.log_analytics_client.workspaces.begin_delete(self.resource_group, self.name, force=self.force)
        except Exception as exc:
            self.fail('Error when deleting workspace {0} - {1}'.format(self.name, exc.message or str(exc)))

    def to_dict(self, workspace):
        result = workspace.as_dict()
        result['sku'] = _camel_to_snake(workspace.sku.name)
        return result

    def list_intelligence_packs(self):
        try:
            response = self.log_analytics_client.intelligence_packs.list(self.resource_group, self.name)
            return [x.as_dict() for x in response]
        except Exception as exc:
            self.fail('Error when listing intelligence packs {0}'.format(exc.message or str(exc)))

    def change_intelligence(self, key, value):
        try:
            if value:
                self.log_analytics_client.intelligence_packs.enable(self.resource_group, self.name, key)
            else:
                self.log_analytics_client.intelligence_packs.disable(self.resource_group, self.name, key)
        except Exception as exc:
            self.fail('Error when changing intelligence pack {0} - {1}'.format(key, exc.message or str(exc)))

    def list_management_groups(self):
        result = []
        try:
            response = self.log_analytics_client.management_groups.list(self.resource_group, self.name)
            while True:
                result.append(response.next().as_dict())
        except StopIteration:
            pass
        except Exception as exc:
            self.fail('Error when listing management groups {0}'.format(exc.message or str(exc)))
        return result

    def list_usages(self):
        result = []
        try:
            response = self.log_analytics_client.usages.list(self.resource_group, self.name)
            while True:
                result.append(response.next().as_dict())
        except StopIteration:
            pass
        except Exception as exc:
            self.fail('Error when listing usages {0}'.format(exc.message or str(exc)))
        return result

    def get_shared_keys(self):
        try:
            return self.log_analytics_client.shared_keys.get_shared_keys(self.resource_group, self.name).as_dict()
        except Exception as exc:
            self.fail('Error when getting shared key {0}'.format(exc.message or str(exc)))