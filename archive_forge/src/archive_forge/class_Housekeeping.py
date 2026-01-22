from __future__ import absolute_import, division, print_function
import re
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.zabbix.plugins.module_utils.base import ZabbixBase
import ansible_collections.community.zabbix.plugins.module_utils.helpers as zabbix_utils
class Housekeeping(ZabbixBase):

    def get_housekeeping(self):
        try:
            return self._zapi.housekeeping.get({'output': 'extend'})
        except Exception as e:
            self._module.fail_json(msg='Failed to get housekeeping setting: %s' % e)

    def check_time_parameter(self, key_name, value):
        match_result = re.match('^[0-9]+[smhdw]$', value)
        if not match_result:
            self._module.fail_json(msg='Invalid value for %s! Please set value like 365d.' % key_name)

    def update_housekeeping(self, current_housekeeping, hk_events_mode, hk_events_trigger, hk_events_service, hk_events_internal, hk_events_discovery, hk_events_autoreg, hk_services_mode, hk_services, hk_audit_mode, hk_audit, hk_sessions_mode, hk_sessions, hk_history_mode, hk_history_global, hk_history, hk_trends_mode, hk_trends_global, hk_trends, compression_status, compress_older):
        try:
            params = {}
            if isinstance(hk_events_mode, bool):
                params['hk_events_mode'] = str(int(hk_events_mode))
            if hk_events_trigger:
                self.check_time_parameter('hk_events_trigger', hk_events_trigger)
                params['hk_events_trigger'] = hk_events_trigger
            if hk_events_service:
                self.check_time_parameter('hk_events_service', hk_events_service)
                params['hk_events_service'] = hk_events_service
            if hk_events_internal:
                self.check_time_parameter('hk_events_internal', hk_events_internal)
                params['hk_events_internal'] = hk_events_internal
            if hk_events_discovery:
                self.check_time_parameter('hk_events_discovery', hk_events_discovery)
                params['hk_events_discovery'] = hk_events_discovery
            if hk_events_autoreg:
                self.check_time_parameter('hk_events_autoreg', hk_events_autoreg)
                params['hk_events_autoreg'] = hk_events_autoreg
            if isinstance(hk_services_mode, bool):
                params['hk_services_mode'] = str(int(hk_services_mode))
            if hk_services:
                self.check_time_parameter('hk_services', hk_services)
                params['hk_services'] = hk_services
            if isinstance(hk_audit_mode, bool):
                params['hk_audit_mode'] = str(int(hk_audit_mode))
            if hk_audit:
                self.check_time_parameter('hk_audit', hk_audit)
                params['hk_audit'] = hk_audit
            if isinstance(hk_sessions_mode, bool):
                params['hk_sessions_mode'] = str(int(hk_sessions_mode))
            if hk_sessions:
                self.check_time_parameter('hk_sessions', hk_sessions)
                params['hk_sessions'] = hk_sessions
            if isinstance(hk_history_mode, bool):
                params['hk_history_mode'] = str(int(hk_history_mode))
            if isinstance(hk_history_global, bool):
                params['hk_history_global'] = str(int(hk_history_global))
            if hk_history:
                self.check_time_parameter('hk_history', hk_history)
                params['hk_history'] = hk_history
            if isinstance(hk_trends_mode, bool):
                params['hk_trends_mode'] = str(int(hk_trends_mode))
            if isinstance(hk_trends_global, bool):
                params['hk_trends_global'] = str(int(hk_trends_global))
            if hk_trends:
                self.check_time_parameter('hk_trends', hk_trends)
                params['hk_trends'] = hk_trends
            if isinstance(compression_status, bool):
                params['compression_status'] = str(int(compression_status))
            if compress_older:
                self.check_time_parameter('compress_older', compress_older)
                params['compress_older'] = compress_older
            future_housekeeping = current_housekeeping.copy()
            future_housekeeping.update(params)
            if future_housekeeping != current_housekeeping:
                if self._module.check_mode:
                    self._module.exit_json(changed=True)
                self._zapi.housekeeping.update(params)
                self._module.exit_json(changed=True, result='Successfully update housekeeping setting')
            else:
                self._module.exit_json(changed=False, result='Housekeeping setting is already up to date')
        except Exception as e:
            self._module.fail_json(msg='Failed to update housekeeping setting, Exception: %s' % e)