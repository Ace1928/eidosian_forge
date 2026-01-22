from __future__ import absolute_import, division, print_function
from ansible_collections.azure.azcollection.plugins.module_utils.azure_rm_common_ext import AzureRMModuleBaseExt
class AzureRMMonitorDiagnosticSetting(AzureRMModuleBaseExt):

    def __init__(self):
        self.module_arg_spec = dict(name=dict(type='str', required=True), resource=dict(type='raw', required=True), storage_account=dict(type='raw'), log_analytics=dict(type='raw'), event_hub=dict(type='dict', options=event_hub_spec), logs=dict(type='list', elements='dict', options=logs_spec), metrics=dict(type='list', elements='dict', options=metrics_spec), state=dict(type='str', default='present', choices=['present', 'absent']))
        self.name = None
        self.resource = None
        self.state = None
        self.parameters = dict()
        self.results = dict(changed=False, state=dict())
        self.to_do = Actions.NoAction
        super(AzureRMMonitorDiagnosticSetting, self).__init__(self.module_arg_spec, required_if=[('state', 'present', ('storage_account', 'log_analytics', 'event_hub'), True), ('state', 'present', ('logs', 'metrics'), True)], supports_tags=False, supports_check_mode=True)

    def exec_module(self, **kwargs):
        for key in self.module_arg_spec:
            if hasattr(self, key):
                setattr(self, key, kwargs[key])
            elif kwargs[key] is not None:
                self.parameters[key] = kwargs[key]
        self.process_parameters()
        old_response = self.get_item()
        if old_response is None or not old_response:
            if self.state == 'present':
                self.to_do = Actions.Create
        elif self.state == 'absent':
            self.to_do = Actions.Delete
        else:
            self.results['compare'] = []
            if not self.idempotency_check(old_response, self.diagnostic_setting_to_dict(self.parameters)):
                self.to_do = Actions.Update
        if self.to_do == Actions.Create or self.to_do == Actions.Update:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.create_update_setting()
        elif self.to_do == Actions.Delete:
            self.results['changed'] = True
            if self.check_mode:
                return self.results
            response = self.delete_setting()
        else:
            self.results['changed'] = False
            response = old_response
        if response is not None:
            self.results['state'] = response
        return self.results

    def process_parameters(self):
        if isinstance(self.resource, dict):
            if '/' not in self.resource.get('type'):
                self.fail("resource type parameter must include namespace, such as 'Microsoft.Network/virtualNetworks'")
            self.resource = resource_id(subscription=self.resource.get('subscription_id', self.subscription_id), resource_group=self.resource.get('resource_group'), namespace=self.resource.get('type').split('/')[0], type=self.resource.get('type').split('/')[1], name=self.resource.get('name'))
        parsed_resource = parse_resource_id(self.resource)
        storage_account = self.parameters.pop('storage_account', None)
        if storage_account:
            if isinstance(storage_account, dict):
                if not storage_account.get('name'):
                    self.fail("storage_account must contain 'name'")
                storage_account_id = resource_id(subscription=storage_account.get('subscription_id', parsed_resource.get('subscription')), resource_group=storage_account.get('resource_group', parsed_resource.get('resource_group')), namespace='Microsoft.Storage', type='storageAccounts', name=storage_account.get('name'))
            else:
                storage_account_id = storage_account
            self.parameters['storage_account_id'] = storage_account_id
        log_analytics = self.parameters.pop('log_analytics', None)
        if log_analytics:
            if isinstance(log_analytics, dict):
                if not log_analytics.get('name'):
                    self.fail("log_analytics must contain 'name'")
                log_analytics_id = resource_id(subscription=log_analytics.get('subscription_id', parsed_resource.get('subscription')), resource_group=log_analytics.get('resource_group', parsed_resource.get('resource_group')), namespace='microsoft.operationalinsights', type='workspaces', name=log_analytics.get('name'))
            else:
                log_analytics_id = log_analytics
            self.parameters['workspace_id'] = log_analytics_id
        event_hub = self.parameters.pop('event_hub', None)
        if event_hub:
            hub_subscription_id = event_hub.get('subscription_id') if event_hub.get('subscription_id') else parsed_resource.get('subscription')
            hub_resource_group = event_hub.get('resource_group') if event_hub.get('resource_group') else parsed_resource.get('resource_group')
            auth_rule_id = resource_id(subscription=hub_subscription_id, resource_group=hub_resource_group, namespace='Microsoft.EventHub', type='namespaces', name=event_hub.get('namespace'), child_type_1='authorizationrules', child_name_1=event_hub.get('policy'))
            self.parameters['event_hub_authorization_rule_id'] = auth_rule_id
            self.parameters['event_hub_name'] = event_hub.get('hub')

    def get_item(self):
        self.log('Get diagnostic setting for {0} in {1}'.format(self.name, self.resource))
        try:
            item = self.monitor_diagnostic_settings_client.diagnostic_settings.get(resource_uri=self.resource, name=self.name)
            return self.diagnostic_setting_to_dict(item)
        except Exception:
            self.log('Did not find diagnostic setting for {0} in {1}'.format(self.name, self.resource))
        return None

    def create_update_setting(self):
        try:
            response = self.monitor_diagnostic_settings_client.diagnostic_settings.create_or_update(resource_uri=self.resource, name=self.name, parameters=self.parameters)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
            return self.diagnostic_setting_to_dict(response)
        except Exception as exc:
            self.fail('Error creating or updating diagnostic setting {0} for resource {1}: {2}'.format(self.name, self.resource, str(exc)))

    def delete_setting(self):
        try:
            response = self.monitor_diagnostic_settings_client.diagnostic_settings.delete(resource_uri=self.resource, name=self.name)
            if isinstance(response, LROPoller):
                response = self.get_poller_result(response)
            return response
        except Exception as exc:
            self.fail('Error deleting diagnostic setting {0} for resource {1}: {2}'.format(self.name, self.resource, str(exc)))

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