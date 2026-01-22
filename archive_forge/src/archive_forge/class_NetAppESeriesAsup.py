from __future__ import absolute_import, division, print_function
import time
from ansible_collections.netapp_eseries.santricity.plugins.module_utils.santricity import NetAppESeriesModule
from ansible.module_utils._text import to_native
class NetAppESeriesAsup(NetAppESeriesModule):
    DAYS_OPTIONS = ['sunday', 'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday']

    def __init__(self):
        ansible_options = dict(state=dict(type='str', required=False, default='enabled', choices=['enabled', 'disabled', 'maintenance_enabled', 'maintenance_disabled']), active=dict(type='bool', required=False, default=True), days=dict(type='list', required=False, aliases=['schedule_days', 'days_of_week'], choices=self.DAYS_OPTIONS), start=dict(type='int', required=False, default=0), end=dict(type='int', required=False, default=24), method=dict(type='str', required=False, choices=['https', 'http', 'email'], default='https'), routing_type=dict(type='str', required=False, choices=['direct', 'proxy', 'script'], default='direct'), proxy=dict(type='dict', required=False, options=dict(host=dict(type='str', required=False), port=dict(type='int', required=False), script=dict(type='str', required=False), username=dict(type='str', required=False), password=dict(type='str', no_log=True, required=False))), email=dict(type='dict', required=False, options=dict(server=dict(type='str', required=False), sender=dict(type='str', required=False), test_recipient=dict(type='str', required=False))), maintenance_duration=dict(type='int', required=False, default=24), maintenance_emails=dict(type='list', required=False), validate=dict(type='bool', require=False, default=False))
        mutually_exclusive = [['host', 'script'], ['port', 'script']]
        required_if = [['method', 'https', ['routing_type']], ['method', 'http', ['routing_type']], ['method', 'email', ['email']], ['state', 'maintenance_enabled', ['maintenance_duration', 'maintenance_emails']]]
        super(NetAppESeriesAsup, self).__init__(ansible_options=ansible_options, web_services_version='02.00.0000.0000', mutually_exclusive=mutually_exclusive, required_if=required_if, supports_check_mode=True)
        args = self.module.params
        self.state = args['state']
        self.active = args['active']
        self.days = args['days']
        self.start = args['start']
        self.end = args['end']
        self.method = args['method']
        self.routing_type = args['routing_type'] if args['routing_type'] else 'none'
        self.proxy = args['proxy']
        self.email = args['email']
        self.maintenance_duration = args['maintenance_duration']
        self.maintenance_emails = args['maintenance_emails']
        self.validate = args['validate']
        if self.validate and self.email and ('test_recipient' not in self.email.keys()):
            self.module.fail_json(msg='test_recipient must be provided for validating email delivery method. Array [%s]' % self.ssid)
        self.check_mode = self.module.check_mode
        if self.start >= self.end:
            self.module.fail_json(msg='The value provided for the start time is invalid. It must be less than the end time.')
        if self.start < 0 or self.start > 23:
            self.module.fail_json(msg='The value provided for the start time is invalid. It must be between 0 and 23.')
        else:
            self.start = self.start * 60
        if self.end < 1 or self.end > 24:
            self.module.fail_json(msg='The value provided for the end time is invalid. It must be between 1 and 24.')
        else:
            self.end = min(self.end * 60, 1439)
        if self.maintenance_duration < 1 or self.maintenance_duration > 72:
            self.module.fail_json(msg='The maintenance duration must be equal to or between 1 and 72 hours.')
        if not self.days:
            self.days = self.DAYS_OPTIONS
        self.url_path_prefix = ''
        if not self.is_embedded() and self.ssid != '0' and (self.ssid.lower() != 'proxy'):
            self.url_path_prefix = 'storage-systems/%s/forward/devmgr/v2/' % self.ssid

    def get_configuration(self):
        try:
            rc, result = self.request(self.url_path_prefix + 'device-asup')
            if not (result['asupCapable'] and result['onDemandCapable']):
                self.module.fail_json(msg='ASUP is not supported on this device. Array Id [%s].' % self.ssid)
            return result
        except Exception as err:
            self.module.fail_json(msg='Failed to retrieve ASUP configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))

    def in_maintenance_mode(self):
        """Determine whether storage device is currently in maintenance mode."""
        results = False
        try:
            rc, key_values = self.request(self.url_path_prefix + 'key-values')
            for key_value in key_values:
                if key_value['key'] == 'ansible_asup_maintenance_email_list':
                    if not self.maintenance_emails:
                        self.maintenance_emails = key_value['value'].split(',')
                elif key_value['key'] == 'ansible_asup_maintenance_stop_time':
                    if time.time() < float(key_value['value']):
                        results = True
        except Exception as error:
            self.module.fail_json(msg='Failed to retrieve maintenance windows information! Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
        return results

    def update_configuration(self):
        config = self.get_configuration()
        update = False
        body = dict()
        if self.state == 'enabled':
            body = dict(asupEnabled=True)
            if not config['asupEnabled']:
                update = True
            if (config['onDemandEnabled'] and config['remoteDiagsEnabled']) != self.active:
                update = True
                body.update(dict(onDemandEnabled=self.active, remoteDiagsEnabled=self.active))
            self.days.sort()
            config['schedule']['daysOfWeek'].sort()
            body['schedule'] = dict(daysOfWeek=self.days, dailyMinTime=self.start, dailyMaxTime=self.end, weeklyMinTime=self.start, weeklyMaxTime=self.end)
            if self.days != config['schedule']['daysOfWeek']:
                update = True
            if self.start != config['schedule']['dailyMinTime'] or self.start != config['schedule']['weeklyMinTime']:
                update = True
            elif self.end != config['schedule']['dailyMaxTime'] or self.end != config['schedule']['weeklyMaxTime']:
                update = True
            if self.method in ['https', 'http']:
                if self.routing_type == 'direct':
                    body['delivery'] = dict(method=self.method, routingType='direct')
                elif self.routing_type == 'proxy':
                    body['delivery'] = dict(method=self.method, proxyHost=self.proxy['host'], proxyPort=self.proxy['port'], routingType='proxyServer')
                    if 'username' in self.proxy.keys():
                        body['delivery'].update({'proxyUserName': self.proxy['username']})
                    if 'password' in self.proxy.keys():
                        body['delivery'].update({'proxyPassword': self.proxy['password']})
                elif self.routing_type == 'script':
                    body['delivery'] = dict(method=self.method, proxyScript=self.proxy['script'], routingType='proxyScript')
            else:
                body['delivery'] = dict(method='smtp', mailRelayServer=self.email['server'], mailSenderAddress=self.email['sender'], routingType='none')
            if config['delivery']['method'] != body['delivery']['method']:
                update = True
            elif config['delivery']['method'] in ['https', 'http']:
                if config['delivery']['routingType'] != body['delivery']['routingType']:
                    update = True
                elif config['delivery']['routingType'] == 'proxyServer':
                    if config['delivery']['proxyHost'] != body['delivery']['proxyHost'] or config['delivery']['proxyPort'] != body['delivery']['proxyPort'] or config['delivery']['proxyUserName'] != body['delivery']['proxyUserName'] or (config['delivery']['proxyPassword'] != body['delivery']['proxyPassword']):
                        update = True
                elif config['delivery']['routingType'] == 'proxyScript':
                    if config['delivery']['proxyScript'] != body['delivery']['proxyScript']:
                        update = True
            elif config['delivery']['method'] == 'smtp' and config['delivery']['mailRelayServer'] != body['delivery']['mailRelayServer'] and (config['delivery']['mailSenderAddress'] != body['delivery']['mailSenderAddress']):
                update = True
            if self.in_maintenance_mode():
                update = True
        elif self.state == 'disabled':
            if config['asupEnabled']:
                body = dict(asupEnabled=False)
                update = True
        else:
            if not config['asupEnabled']:
                self.module.fail_json(msg='AutoSupport must be enabled before enabling or disabling maintenance mode. Array [%s].' % self.ssid)
            if self.in_maintenance_mode() or self.state == 'maintenance_enabled':
                update = True
        if update and (not self.check_mode):
            if self.state == 'maintenance_enabled':
                try:
                    rc, response = self.request(self.url_path_prefix + 'device-asup/maintenance-window', method='POST', data=dict(maintenanceWindowEnabled=True, duration=self.maintenance_duration, emailAddresses=self.maintenance_emails))
                except Exception as error:
                    self.module.fail_json(msg='Failed to enabled ASUP maintenance window. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
                try:
                    rc, response = self.request(self.url_path_prefix + 'key-values/ansible_asup_maintenance_email_list', method='POST', data=','.join(self.maintenance_emails))
                    rc, response = self.request(self.url_path_prefix + 'key-values/ansible_asup_maintenance_stop_time', method='POST', data=str(time.time() + 60 * 60 * self.maintenance_duration))
                except Exception as error:
                    self.module.fail_json(msg='Failed to store maintenance information. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
            elif self.state == 'maintenance_disabled':
                try:
                    rc, response = self.request(self.url_path_prefix + 'device-asup/maintenance-window', method='POST', data=dict(maintenanceWindowEnabled=False, emailAddresses=self.maintenance_emails))
                except Exception as error:
                    self.module.fail_json(msg='Failed to disable ASUP maintenance window. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
                try:
                    rc, response = self.request(self.url_path_prefix + 'key-values/ansible_asup_maintenance_email_list', method='DELETE')
                    rc, response = self.request(self.url_path_prefix + 'key-values/ansible_asup_maintenance_stop_time', method='DELETE')
                except Exception as error:
                    self.module.fail_json(msg='Failed to store maintenance information. Array [%s]. Error [%s].' % (self.ssid, to_native(error)))
            else:
                if body['asupEnabled'] and self.validate:
                    validate_body = dict(delivery=body['delivery'])
                    if self.email:
                        validate_body['mailReplyAddress'] = self.email['test_recipient']
                    try:
                        rc, response = self.request(self.url_path_prefix + 'device-asup/verify-config', timeout=600, method='POST', data=validate_body)
                    except Exception as err:
                        self.module.fail_json(msg='Failed to validate ASUP configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
                try:
                    rc, response = self.request(self.url_path_prefix + 'device-asup', method='POST', data=body)
                except Exception as err:
                    self.module.fail_json(msg='Failed to change ASUP configuration! Array Id [%s]. Error [%s].' % (self.ssid, to_native(err)))
        return update

    def apply(self):
        update = self.update_configuration()
        cfg = self.get_configuration()
        if update:
            self.module.exit_json(msg='The ASUP settings have been updated.', changed=update, asup=cfg['asupEnabled'], active=cfg['onDemandEnabled'], cfg=cfg)
        else:
            self.module.exit_json(msg='No ASUP changes required.', changed=update, asup=cfg['asupEnabled'], active=cfg['onDemandEnabled'], cfg=cfg)