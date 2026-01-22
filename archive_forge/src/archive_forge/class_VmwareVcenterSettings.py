from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.vmware.plugins.module_utils.vmware import PyVmomi, option_diff, vmware_argument_spec
from ansible.module_utils._text import to_native
class VmwareVcenterSettings(PyVmomi):
    """Manage settings for a vCenter server"""

    def __init__(self, module):
        super(VmwareVcenterSettings, self).__init__(module)
        if not self.is_vcenter():
            self.module.fail_json(msg='You have to connect to a vCenter server!')
        self.option_manager = self.content.setting

    def get_default_setting_value(self, setting_key):
        return self.option_manager.QueryOptions(name=setting_key)[0].value

    def ensure(self):
        """Manage settings for a vCenter server"""
        result = dict(changed=False, msg='')
        message = ''
        db_max_connections = self.params['database'].get('max_connections')
        db_task_cleanup = self.params['database'].get('task_cleanup')
        db_task_retention = self.params['database'].get('task_retention')
        db_event_cleanup = self.params['database'].get('event_cleanup')
        db_event_retention = self.params['database'].get('event_retention')
        runtime_unique_id = self.get_default_setting_value('instance.id')
        runtime_managed_address = self.get_default_setting_value('VirtualCenter.ManagedIP')
        runtime_server_name = self.get_default_setting_value('VirtualCenter.InstanceName')
        if self.params['runtime_settings']:
            if self.params['runtime_settings'].get('unique_id') is not None:
                runtime_unique_id = self.params['runtime_settings'].get('unique_id')
            if self.params['runtime_settings'].get('managed_address') is not None:
                runtime_managed_address = self.params['runtime_settings'].get('managed_address')
            if self.params['runtime_settings'].get('vcenter_server_name') is not None:
                runtime_server_name = self.params['runtime_settings'].get('vcenter_server_name')
        directory_timeout = self.params['user_directory'].get('timeout')
        directory_query_limit = self.params['user_directory'].get('query_limit')
        directory_query_limit_size = self.params['user_directory'].get('query_limit_size')
        directory_validation = self.params['user_directory'].get('validation')
        directory_validation_period = self.params['user_directory'].get('validation_period')
        mail = self.params.get('mail') or {'mail': {'server': '', 'sender': ''}}
        mail_server = mail.get('server', '')
        mail_sender = mail.get('sender', '')
        snmp_receiver_1_url = self.params['snmp_receivers'].get('snmp_receiver_1_url')
        snmp_receiver_1_enabled = self.params['snmp_receivers'].get('snmp_receiver_1_enabled')
        snmp_receiver_1_port = self.params['snmp_receivers'].get('snmp_receiver_1_port')
        snmp_receiver_1_community = self.params['snmp_receivers'].get('snmp_receiver_1_community')
        snmp_receiver_2_url = self.params['snmp_receivers'].get('snmp_receiver_2_url')
        snmp_receiver_2_enabled = self.params['snmp_receivers'].get('snmp_receiver_2_enabled')
        snmp_receiver_2_port = self.params['snmp_receivers'].get('snmp_receiver_2_port')
        snmp_receiver_2_community = self.params['snmp_receivers'].get('snmp_receiver_2_community')
        snmp_receiver_3_url = self.params['snmp_receivers'].get('snmp_receiver_3_url')
        snmp_receiver_3_enabled = self.params['snmp_receivers'].get('snmp_receiver_3_enabled')
        snmp_receiver_3_port = self.params['snmp_receivers'].get('snmp_receiver_3_port')
        snmp_receiver_3_community = self.params['snmp_receivers'].get('snmp_receiver_3_community')
        snmp_receiver_4_url = self.params['snmp_receivers'].get('snmp_receiver_4_url')
        snmp_receiver_4_enabled = self.params['snmp_receivers'].get('snmp_receiver_4_enabled')
        snmp_receiver_4_port = self.params['snmp_receivers'].get('snmp_receiver_4_port')
        snmp_receiver_4_community = self.params['snmp_receivers'].get('snmp_receiver_4_community')
        timeout_normal_operations = self.params['timeout_settings'].get('normal_operations')
        timeout_long_operations = self.params['timeout_settings'].get('long_operations')
        logging_options = self.params.get('logging_options')
        changed = False
        changed_list = []
        result['db_max_connections'] = db_max_connections
        result['db_task_cleanup'] = db_task_cleanup
        result['db_task_retention'] = db_task_retention
        result['db_event_cleanup'] = db_event_cleanup
        result['db_event_retention'] = db_event_retention
        result['runtime_unique_id'] = runtime_unique_id
        result['runtime_managed_address'] = runtime_managed_address
        result['runtime_server_name'] = runtime_server_name
        result['directory_timeout'] = directory_timeout
        result['directory_query_limit'] = directory_query_limit
        result['directory_query_limit_size'] = directory_query_limit_size
        result['directory_validation'] = directory_validation
        result['directory_validation_period'] = directory_validation_period
        result['mail_server'] = mail_server
        result['mail_sender'] = mail_sender
        result['timeout_normal_operations'] = timeout_normal_operations
        result['timeout_long_operations'] = timeout_long_operations
        result['logging_options'] = logging_options
        change_option_list = []
        diff_config = dict(before={}, after={})
        for key in result.keys():
            if key != 'changed' and key != 'msg':
                diff_config['before'][key] = result[key]
                diff_config['after'][key] = result[key]
        for n in range(1, 5):
            exec("diff_config['before']['snmp_receiver_%s_url'] = snmp_receiver_%s_url" % (n, n))
            exec("diff_config['before']['snmp_receiver_%s_enabled'] = snmp_receiver_%s_enabled" % (n, n))
            exec("diff_config['before']['snmp_receiver_%s_port'] = snmp_receiver_%s_port" % (n, n))
            exec("diff_config['before']['snmp_receiver_%s_community'] = snmp_receiver_%s_community" % (n, n))
            exec("diff_config['after']['snmp_receiver_%s_url'] = snmp_receiver_%s_url" % (n, n))
            exec("diff_config['after']['snmp_receiver_%s_enabled'] = snmp_receiver_%s_enabled" % (n, n))
            exec("diff_config['after']['snmp_receiver_%s_port'] = snmp_receiver_%s_port" % (n, n))
            exec("diff_config['after']['snmp_receiver_%s_community'] = snmp_receiver_%s_community" % (n, n))
        result['diff'] = {}
        advanced_settings = self.params['advanced_settings']
        changed_advanced_settings = option_diff(advanced_settings, self.option_manager.setting, False)
        if changed_advanced_settings:
            changed = True
            change_option_list += changed_advanced_settings
        for advanced_setting in advanced_settings:
            result[advanced_setting] = advanced_settings[advanced_setting]
            diff_config['before'][advanced_setting] = result[advanced_setting]
            diff_config['after'][advanced_setting] = result[advanced_setting]
        for setting in self.option_manager.setting:
            if setting.key == 'VirtualCenter.MaxDBConnection' and setting.value != db_max_connections:
                changed = True
                changed_list.append('DB max connections')
                result['db_max_connections_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='VirtualCenter.MaxDBConnection', value=db_max_connections))
                diff_config['before']['db_max_connections'] = setting.value
            if setting.key == 'task.maxAgeEnabled' and setting.value != db_task_cleanup:
                changed = True
                changed_list.append('DB task cleanup')
                result['db_task_cleanup_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='task.maxAgeEnabled', value=db_task_cleanup))
                diff_config['before']['db_task_cleanup'] = setting.value
            if setting.key == 'task.maxAge' and setting.value != db_task_retention:
                changed = True
                changed_list.append('DB task retention')
                result['db_task_retention_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='task.maxAge', value=db_task_retention))
                diff_config['before']['db_task_retention'] = setting.value
            if setting.key == 'event.maxAgeEnabled' and setting.value != db_event_cleanup:
                changed = True
                changed_list.append('DB event cleanup')
                result['db_event_cleanup_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='event.maxAgeEnabled', value=db_event_cleanup))
                diff_config['before']['db_event_cleanup'] = setting.value
            if setting.key == 'event.maxAge' and setting.value != db_event_retention:
                changed = True
                changed_list.append('DB event retention')
                result['db_event_retention_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='event.maxAge', value=db_event_retention))
                diff_config['before']['db_event_retention'] = setting.value
            if setting.key == 'instance.id' and setting.value != runtime_unique_id:
                changed = True
                changed_list.append('Instance ID')
                result['runtime_unique_id_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='instance.id', value=runtime_unique_id))
                diff_config['before']['runtime_unique_id'] = setting.value
            if setting.key == 'VirtualCenter.ManagedIP' and setting.value != runtime_managed_address:
                changed = True
                changed_list.append('Managed IP')
                result['runtime_managed_address_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='VirtualCenter.ManagedIP', value=runtime_managed_address))
                diff_config['before']['runtime_managed_address'] = setting.value
            if setting.key == 'VirtualCenter.InstanceName' and setting.value != runtime_server_name:
                changed = True
                changed_list.append('Server name')
                result['runtime_server_name_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='VirtualCenter.InstanceName', value=runtime_server_name))
                diff_config['before']['runtime_server_name'] = setting.value
            if setting.key == 'ads.timeout' and setting.value != directory_timeout:
                changed = True
                changed_list.append('Directory timeout')
                result['directory_timeout_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='ads.timeout', value=directory_timeout))
                diff_config['before']['directory_timeout'] = setting.value
            if setting.key == 'ads.maxFetchEnabled' and setting.value != directory_query_limit:
                changed = True
                changed_list.append('Query limit')
                result['directory_query_limit_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='ads.maxFetchEnabled', value=directory_query_limit))
                diff_config['before']['directory_query_limit'] = setting.value
            if setting.key == 'ads.maxFetch' and setting.value != directory_query_limit_size:
                changed = True
                changed_list.append('Query limit size')
                result['directory_query_limit_size_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='ads.maxFetch', value=directory_query_limit_size))
                diff_config['before']['directory_query_limit_size'] = setting.value
            if setting.key == 'ads.checkIntervalEnabled' and setting.value != directory_validation:
                changed = True
                changed_list.append('Validation')
                result['directory_validation_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='ads.checkIntervalEnabled', value=directory_validation))
                diff_config['before']['directory_validation'] = setting.value
            if setting.key == 'ads.checkInterval' and setting.value != directory_validation_period:
                changed = True
                changed_list.append('Validation period')
                result['directory_validation_period_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='ads.checkInterval', value=directory_validation_period))
                diff_config['before']['directory_validation_period'] = setting.value
            if setting.key == 'mail.smtp.server' and setting.value != mail_server:
                changed = True
                changed_list.append('Mail server')
                result['mail_server_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='mail.smtp.server', value=mail_server))
                diff_config['before']['mail_server'] = setting.value
            if setting.key == 'mail.sender' and setting.value != mail_sender:
                changed = True
                changed_list.append('Mail sender')
                result['mail_sender_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='mail.sender', value=mail_sender))
                diff_config['before']['mail_sender'] = setting.value
            if setting.key == 'snmp.receiver.1.enabled' and setting.value != snmp_receiver_1_enabled:
                changed = True
                changed_list.append('SNMP-1-enabled')
                result['snmp_1_enabled_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.1.enabled', value=snmp_receiver_1_enabled))
                diff_config['before']['snmp_receiver_1_enabled'] = setting.value
            if setting.key == 'snmp.receiver.1.name' and setting.value != snmp_receiver_1_url:
                changed = True
                changed_list.append('SNMP-1-name')
                result['snmp_1_url_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.1.name', value=snmp_receiver_1_url))
                diff_config['before']['snmp_receiver_1_url'] = setting.value
            if setting.key == 'snmp.receiver.1.port' and setting.value != snmp_receiver_1_port:
                changed = True
                changed_list.append('SNMP-1-port')
                result['snmp_receiver_1_port_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.1.port', value=snmp_receiver_1_port))
                diff_config['before']['snmp_receiver_1_port'] = setting.value
            if setting.key == 'snmp.receiver.1.community' and setting.value != snmp_receiver_1_community:
                changed = True
                changed_list.append('SNMP-1-community')
                result['snmp_1_community_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.1.community', value=snmp_receiver_1_community))
                diff_config['before']['snmp_receiver_1_community'] = setting.value
            if setting.key == 'snmp.receiver.2.enabled' and setting.value != snmp_receiver_2_enabled:
                changed = True
                changed_list.append('SNMP-2-enabled')
                result['snmp_2_enabled_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.2.enabled', value=snmp_receiver_2_enabled))
                diff_config['before']['snmp_receiver_2_enabled'] = setting.value
            if setting.key == 'snmp.receiver.2.name' and setting.value != snmp_receiver_2_url:
                changed = True
                changed_list.append('SNMP-2-name')
                result['snmp_2_url_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.2.name', value=snmp_receiver_2_url))
                diff_config['before']['snmp_receiver_2_url'] = setting.value
            if setting.key == 'snmp.receiver.2.port' and setting.value != snmp_receiver_2_port:
                changed = True
                changed_list.append('SNMP-2-port')
                result['snmp_receiver_2_port_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.2.port', value=snmp_receiver_2_port))
                diff_config['before']['snmp_receiver_2_port'] = setting.value
            if setting.key == 'snmp.receiver.2.community' and setting.value != snmp_receiver_2_community:
                changed = True
                changed_list.append('SNMP-2-community')
                result['snmp_2_community_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.2.community', value=snmp_receiver_2_community))
                diff_config['before']['snmp_receiver_2_community'] = setting.value
            if setting.key == 'snmp.receiver.3.enabled' and setting.value != snmp_receiver_3_enabled:
                changed = True
                changed_list.append('SNMP-3-enabled')
                result['snmp_3_enabled_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.3.enabled', value=snmp_receiver_3_enabled))
                diff_config['before']['snmp_receiver_3_enabled'] = setting.value
            if setting.key == 'snmp.receiver.3.name' and setting.value != snmp_receiver_3_url:
                changed = True
                changed_list.append('SNMP-3-name')
                result['snmp_3_url_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.3.name', value=snmp_receiver_3_url))
                diff_config['before']['snmp_receiver_3_url'] = setting.value
            if setting.key == 'snmp.receiver.3.port' and setting.value != snmp_receiver_3_port:
                changed = True
                changed_list.append('SNMP-3-port')
                result['snmp_receiver_3_port_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.3.port', value=snmp_receiver_3_port))
                diff_config['before']['snmp_receiver_3_port'] = setting.value
            if setting.key == 'snmp.receiver.3.community' and setting.value != snmp_receiver_3_community:
                changed = True
                changed_list.append('SNMP-3-community')
                result['snmp_3_community_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.3.community', value=snmp_receiver_3_community))
                diff_config['before']['snmp_receiver_3_community'] = setting.value
            if setting.key == 'snmp.receiver.4.enabled' and setting.value != snmp_receiver_4_enabled:
                changed = True
                changed_list.append('SNMP-4-enabled')
                result['snmp_4_enabled_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.4.enabled', value=snmp_receiver_4_enabled))
                diff_config['before']['snmp_receiver_4_enabled'] = setting.value
            if setting.key == 'snmp.receiver.4.name' and setting.value != snmp_receiver_4_url:
                changed = True
                changed_list.append('SNMP-4-name')
                result['snmp_4_url_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.4.name', value=snmp_receiver_4_url))
                diff_config['before']['snmp_receiver_4_url'] = setting.value
            if setting.key == 'snmp.receiver.4.port' and setting.value != snmp_receiver_4_port:
                changed = True
                changed_list.append('SNMP-4-port')
                result['snmp_receiver_4_port_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.4.port', value=snmp_receiver_4_port))
                diff_config['before']['snmp_receiver_4_port'] = setting.value
            if setting.key == 'snmp.receiver.4.community' and setting.value != snmp_receiver_4_community:
                changed = True
                changed_list.append('SNMP-4-community')
                result['snmp_4_community_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='snmp.receiver.4.community', value=snmp_receiver_4_community))
                diff_config['before']['snmp_receiver_4_community'] = setting.value
            if setting.key == 'client.timeout.normal' and setting.value != timeout_normal_operations:
                changed = True
                changed_list.append('Timeout normal')
                result['timeout_normal_operations_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='client.timeout.normal', value=timeout_normal_operations))
                diff_config['before']['timeout_normal_operations'] = setting.value
            if setting.key == 'client.timeout.long' and setting.value != timeout_long_operations:
                changed = True
                changed_list.append('Timout long')
                result['timeout_long_operations_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='client.timeout.long', value=timeout_long_operations))
                diff_config['before']['timeout_long_operations'] = setting.value
            if setting.key == 'log.level' and setting.value != logging_options:
                changed = True
                changed_list.append('Logging')
                result['logging_options_previous'] = setting.value
                change_option_list.append(vim.option.OptionValue(key='log.level', value=logging_options))
                diff_config['before']['logging_options'] = setting.value
            for advanced_setting in changed_advanced_settings:
                if setting.key == advanced_setting.key and setting.value != advanced_setting.value:
                    changed_list.append(advanced_setting.key)
                    result[advanced_setting.key + '_previous'] = advanced_setting.value
                    diff_config['before'][advanced_setting.key] = advanced_setting.value
        for advanced_setting in changed_advanced_settings:
            if advanced_setting.key not in changed_list:
                changed_list.append(advanced_setting.key)
                result[advanced_setting.key + '_previous'] = 'N/A'
                diff_config['before'][advanced_setting.key] = 'N/A'
        if changed:
            if self.module.check_mode:
                changed_suffix = ' would be changed'
            else:
                changed_suffix = ' changed'
            if len(changed_list) > 2:
                message = ', '.join(changed_list[:-1]) + ', and ' + str(changed_list[-1])
            elif len(changed_list) == 2:
                message = ' and '.join(changed_list)
            elif len(changed_list) == 1:
                message = changed_list[0]
            message += changed_suffix
            if not self.module.check_mode:
                try:
                    self.option_manager.UpdateOptions(changedValue=change_option_list)
                except (vmodl.fault.SystemError, vmodl.fault.InvalidArgument) as invalid_argument:
                    self.module.fail_json(msg='Failed to update option(s) as one or more OptionValue contains an invalid value: %s' % to_native(invalid_argument.msg))
                except vim.fault.InvalidName as invalid_name:
                    self.module.fail_json(msg='Failed to update option(s) as one or more OptionValue objects refers to a non-existent option : %s' % to_native(invalid_name.msg))
        else:
            message = 'vCenter settings already configured properly'
        result['changed'] = changed
        result['msg'] = message
        result['diff']['before'] = OrderedDict(sorted(diff_config['before'].items()))
        result['diff']['after'] = OrderedDict(sorted(diff_config['after'].items()))
        self.module.exit_json(**result)