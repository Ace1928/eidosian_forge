from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
class AzureRMMonitorDiagnosticSettingInfo(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str'), resource=dict(type='raw', required=True))
        self.results = dict(changed=False, settings=[])
        self.name = None
        self.resource = None
        super(AzureRMMonitorDiagnosticSettingInfo, self).__init__(self.module_arg_spec, supports_check_mode=True, supports_tags=False, facts_module=True)

    def exec_module(self, **kwargs):
        for key in list(self.module_arg_spec.keys()):
            setattr(self, key, kwargs[key])
        self.process_parameters()
        if self.name is not None:
            self.results['settings'] = self.get_item()
        else:
            self.results['settings'] = self.list_items()
        return self.results

    def process_parameters(self):
        if isinstance(self.resource, dict):
            if '/' not in self.resource.get('type'):
                self.fail("resource type parameter must include namespace, such as 'Microsoft.Network/virtualNetworks'")
            self.resource = resource_id(subscription=self.resource.get('subscription_id', self.subscription_id), resource_group=self.resource.get('resource_group'), namespace=self.resource.get('type').split('/')[0], type=self.resource.get('type').split('/')[1], name=self.resource.get('name'))

    def get_item(self):
        self.log('Get diagnostic setting for {0} in {1}'.format(self.name, self.resource))
        try:
            item = self.monitor_diagnostic_settings_client.diagnostic_settings.get(resource_uri=self.resource, name=self.name)
            return [self.diagnostic_setting_to_dict(item)]
        except Exception:
            self.log('Could not get diagnostic setting for {0} in {1}'.format(self.name, self.resource))
        return []

    def list_items(self):
        self.log('List all diagnostic settings in {0}'.format(self.resource))
        try:
            items = self.monitor_diagnostic_settings_client.diagnostic_settings.list(resource_uri=self.resource)
            items = [self.diagnostic_setting_to_dict(item) for item in items]
            items = sorted(items, key=lambda d: d['name'])
            return items
        except Exception as exc:
            self.fail('Failed to list all diagnostic settings in {0}: {1}'.format(self.resource, str(exc)))

    def diagnostic_setting_to_dict(self, diagnostic_setting):
        setting_dict = diagnostic_setting if isinstance(diagnostic_setting, dict) else diagnostic_setting.as_dict()
        result = dict(id=setting_dict.get('id'), name=setting_dict.get('name'), event_hub=self.event_hub_dict(setting_dict), storage_account=self.storage_dict(setting_dict.get('storage_account_id')), log_analytics=self.log_analytics_dict(setting_dict.get('workspace_id')), logs=[self.log_config_to_dict(log) for log in setting_dict.get('logs', [])], metrics=[self.metric_config_to_dict(metric) for metric in setting_dict.get('metrics', [])])
        return self.remove_disabled_config(result)

    def remove_disabled_config(self, diagnostic_setting):
        diagnostic_setting['logs'] = [log for log in diagnostic_setting.get('logs', []) if log.get('enabled')]
        diagnostic_setting['metrics'] = [metric for metric in diagnostic_setting.get('metrics', []) if metric.get('enabled')]
        return diagnostic_setting

    def event_hub_dict(self, setting_dict):
        auth_rule_id = setting_dict.get('event_hub_authorization_rule_id')
        if auth_rule_id:
            parsed_rule_id = parse_resource_id(auth_rule_id)
            return dict(id=resource_id(subscription=parsed_rule_id.get('subscription'), resource_group=parsed_rule_id.get('resource_group'), namespace=parsed_rule_id.get('namespace'), type=parsed_rule_id.get('type'), name=parsed_rule_id.get('name')), namespace=parsed_rule_id.get('name'), hub=setting_dict.get('event_hub_name'), policy=parsed_rule_id.get('resource_name'))
        return None

    def storage_dict(self, storage_account_id):
        if storage_account_id:
            return dict(id=storage_account_id)
        return None

    def log_analytics_dict(self, workspace_id):
        if workspace_id:
            return dict(id=workspace_id)
        return None

    def log_config_to_dict(self, log_config):
        return dict(category=log_config.get('category'), category_group=log_config.get('category_group'), enabled=log_config.get('enabled'), retention_policy=self.retention_policy_to_dict(log_config.get('retention_policy')))

    def metric_config_to_dict(self, metric_config):
        return dict(category=metric_config.get('category'), enabled=metric_config.get('enabled'), retention_policy=self.retention_policy_to_dict(metric_config.get('retention_policy')))

    def retention_policy_to_dict(self, policy):
        if policy:
            return dict(days=policy.get('days'), enabled=policy.get('enabled'))
        return None