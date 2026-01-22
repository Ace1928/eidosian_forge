from __future__ import absolute_import, division, print_function
from ansible.module_utils.common.dict_transformations import _camel_to_snake
class AzureRMLogAnalyticsWorkspaceInfo(AzureRMModuleBase):

    def __init__(self):
        self.module_arg_spec = dict(resource_group=dict(type='str', required=True), name=dict(type='str'), tags=dict(type='list', elements='str'), show_shared_keys=dict(type='bool'), show_intelligence_packs=dict(type='bool'), show_usages=dict(type='bool'), show_management_groups=dict(type='bool'))
        self.results = dict(changed=False, workspaces=[])
        self.resource_group = None
        self.name = None
        self.tags = None
        self.show_intelligence_packs = None
        self.show_shared_keys = None
        self.show_usages = None
        self.show_management_groups = None
        super(AzureRMLogAnalyticsWorkspaceInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        is_old_facts = self.module._name == 'azure_rm_loganalyticsworkspace_facts'
        if is_old_facts:
            self.module.deprecate("The 'azure_rm_loganalyticsworkspace_facts' module has been renamed to 'azure_rm_loganalyticsworkspace_info'", version=(2.9,))
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        if self.name:
            item = self.get_workspace()
            response = [item] if item else []
        else:
            response = self.list_by_resource_group()
        self.results['workspaces'] = [self.to_dict(x) for x in response if self.has_tags(x.tags, self.tags)]
        return self.results

    def get_workspace(self):
        try:
            return self.log_analytics_client.workspaces.get(self.resource_group, self.name)
        except ResourceNotFoundError:
            pass
        return None

    def list_by_resource_group(self):
        try:
            return self.log_analytics_client.resource_group.list(self.resource_group)
        except Exception:
            pass
        return []

    def list_intelligence_packs(self):
        try:
            response = self.log_analytics_client.intelligence_packs.list(self.resource_group, self.name)
            return [x.as_dict() for x in response]
        except Exception as exc:
            self.fail('Error when listing intelligence packs {0}'.format(exc.message or str(exc)))

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

    def to_dict(self, workspace):
        result = workspace.as_dict()
        result['sku'] = _camel_to_snake(workspace.sku.name)
        if self.show_intelligence_packs:
            result['intelligence_packs'] = self.list_intelligence_packs()
        if self.show_management_groups:
            result['management_groups'] = self.list_management_groups()
        if self.show_shared_keys:
            result['shared_keys'] = self.get_shared_keys()
        if self.show_usages:
            result['usages'] = self.list_usages()
        return result