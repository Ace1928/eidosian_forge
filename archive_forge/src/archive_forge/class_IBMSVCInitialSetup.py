from __future__ import absolute_import, division, print_function
from traceback import format_exc
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import IBMSVCRestApi, svc_argument_spec, get_logger
from ansible.module_utils._text import to_native
class IBMSVCInitialSetup(object):

    def __init__(self):
        argument_spec = svc_argument_spec()
        argument_spec.update(dict(system_name=dict(type='str'), dnsname=dict(type='list', elements='str'), dnsip=dict(type='list', elements='str'), ntpip=dict(type='str'), time=dict(type='str'), timezone=dict(type='str'), license_key=dict(type='list', elements='str', no_log=True), remote=dict(type='int'), virtualization=dict(type='int'), flash=dict(type='int'), compression=dict(type='int'), cloud=dict(type='int'), easytier=dict(type='int'), physical_flash=dict(type='str', default='off', choices=['on', 'off']), encryption=dict(type='str', choices=['on', 'off'])))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, log_path)
        self.log = log.info
        self.system_data = ''
        self.changed = False
        self.message = ''
        self.systemname = self.module.params.get('system_name', '')
        self.dnsname = self.module.params.get('dnsname', '')
        self.dnsip = self.module.params.get('dnsip', '')
        self.ntpip = self.module.params.get('ntpip', '')
        self.time = self.module.params.get('time', '')
        self.timezone = self.module.params.get('timezone', '')
        self.license_key = self.module.params.get('license_key', '')
        self.remote = self.module.params.get('remote', '')
        self.virtualization = self.module.params.get('virtualization', '')
        self.compression = self.module.params.get('compression', '')
        self.flash = self.module.params.get('flash', '')
        self.cloud = self.module.params.get('cloud', '')
        self.easytier = self.module.params.get('easytier', '')
        self.physical_flash = self.module.params.get('physical_flash', '')
        self.encryption = self.module.params.get('encryption', '')
        self.restapi = IBMSVCRestApi(module=self.module, clustername=self.module.params['clustername'], domain=self.module.params['domain'], username=self.module.params['username'], password=self.module.params['password'], validate_certs=self.module.params['validate_certs'], log_path=log_path, token=self.module.params['token'])

    def basic_checks(self):
        if self.time and self.ntpip:
            self.module.fail_json(msg='Either NTP or time should be given')
        if self.dnsname and self.dnsip:
            if len(self.dnsname) != len(self.dnsip):
                self.module.fail_json(msg='To configure DNS, DNS IP and DNS server name must be given.')

    def get_system_info(self):
        self.log('Entering function get_system_info')
        self.system_data = self.restapi.svc_obj_info(cmd='lssystem', cmdopts=None, cmdargs=None)
        return self.system_data

    def systemname_update(self):
        cmd = 'chsystem'
        cmdopts = {}
        cmdopts['name'] = self.systemname
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.changed = True
        self.log('System Name: %s updated', cmdopts)
        self.message += ' System name [%s] updated.' % self.systemname

    def ntp_update(self, ip):
        cmd = 'chsystem'
        cmdopts = {}
        cmdopts['ntpip'] = ip
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.changed = True
        self.log('NTP IP: %s updated', cmdopts)
        if self.ntpip:
            self.message += ' NTP IP [%s] updated.' % self.ntpip

    def systemtime_update(self):
        cmd = 'setsystemtime'
        cmdopts = {}
        cmdopts['time'] = self.time
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.changed = True
        self.log('Time: %s updated', self.time)
        self.message += ' Time [%s] updated.' % self.time

    def timezone_update(self):
        cmd = 'settimezone'
        cmdopts = {}
        cmdopts['timezone'] = self.timezone
        self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.changed = True
        self.log('Properties: Time zone %s updated', self.timezone)
        self.message += ' Timezone [%s] updated.' % self.timezone

    def system_update(self, data):
        name_change_required = False
        ntp_change_required = False
        time_change_required = False
        timezone_change_required = False
        tz = (None, None)
        if self.module.check_mode:
            self.changed = True
            return
        if self.systemname and self.systemname != data['name']:
            self.log('Name change detected')
            name_change_required = True
        if self.ntpip and self.ntpip != data['cluster_ntp_IP_address']:
            self.log('NTP change detected')
            ntp_change_required = True
        if self.time and data['cluster_ntp_IP_address'] is not None:
            self.log('TIME change detected, clearing NTP IP')
            ntp_change_required = True
        if self.time:
            self.log('TIME change detected')
            time_change_required = True
        if data['time_zone']:
            tz = data['time_zone'].split(' ', 1)
        if self.timezone and tz[0] != self.timezone:
            timezone_change_required = True
        if name_change_required:
            self.systemname_update()
        if ntp_change_required:
            self.log("updating system properties '%s, %s'", self.systemname, self.ntpip)
            if self.ntpip:
                ip = self.ntpip
            if self.time and ntp_change_required:
                ip = '0.0.0.0'
            self.ntp_update(ip)
        if time_change_required:
            self.systemtime_update()
        if timezone_change_required:
            self.timezone_update()

    def get_existing_dnsservers(self):
        merged_result = []
        data = self.restapi.svc_obj_info(cmd='lsdnsserver', cmdopts=None, cmdargs=None)
        if isinstance(data, list):
            for d in data:
                merged_result.append(d)
        else:
            merged_result = data
        return merged_result

    def dns_configure(self):
        dns_add_remove = False
        modify = {}
        existing_dns = {}
        existing_dns_server = []
        existing_dns_ip = []
        if self.module.check_mode:
            self.changed = True
            return
        dns_data = self.get_existing_dnsservers()
        self.log('dns_data=%s', dns_data)
        if self.dnsip and self.dnsname or (self.dnsip == '' and self.dnsname == ''):
            for server in dns_data:
                existing_dns_server.append(server['name'])
                existing_dns_ip.append(server['IP_address'])
                existing_dns[server['name']] = server['IP_address']
            for name, ip in zip(self.dnsname, self.dnsip):
                if name == 'None':
                    self.log(' Empty DNS configuration is provided.')
                    return
                if name in existing_dns:
                    if existing_dns[name] != ip:
                        self.log('update, diff IP.')
                        modify[name] = ip
                    else:
                        self.log('no update, same IP.')
            if set(existing_dns_server).symmetric_difference(set(self.dnsname)):
                dns_add_remove = True
        if modify:
            for item in modify:
                self.restapi.svc_run_command('chdnsserver', {'ip': modify[item]}, [item])
            self.changed = True
            self.message += ' DNS %s modified.' % modify
        if dns_add_remove:
            to_be_added, to_be_removed = (False, False)
            to_be_removed = list(set(existing_dns_server) - set(self.dnsname))
            if to_be_removed:
                for item in to_be_removed:
                    self.restapi.svc_run_command('rmdnsserver', None, [item])
                    self.changed = True
                self.message += ' DNS server %s removed.' % to_be_removed
            to_be_added = list(set(self.dnsname) - set(existing_dns_server))
            to_be_added_ip = list(set(self.dnsip) - set(existing_dns_ip))
            if any(to_be_added):
                for dns_name, dns_ip in zip(to_be_added, to_be_added_ip):
                    if dns_name:
                        self.log('%s %s', dns_name, dns_ip)
                        self.restapi.svc_run_command('mkdnsserver', {'name': dns_name, 'ip': dns_ip}, cmdargs=None)
                        self.changed = True
                self.message += ' DNS server %s added.' % to_be_added
        elif not modify:
            self.log('No DNS Changes')

    def license_probe(self):
        props = []
        cmd = 'lslicense'
        cmdopts = {}
        data = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        if self.remote and int(data['license_remote']) != self.remote:
            props += ['remote']
        if self.virtualization and int(data['license_virtualization']) != self.virtualization:
            props += ['virtualization']
        if self.compression:
            if self.system_data['product_name'] == 'IBM Storwize V7000' or self.system_data['product_name'] == 'IBM FlashSystem 7200':
                if int(data['license_compression_enclosures']) != self.compression:
                    self.log('license_compression_enclosure=%d', int(data['license_compression_enclosures']))
                    props += ['compression']
            elif int(data['license_compression_capacity']) != self.compression:
                self.log('license_compression_capacity=%d', int(data['license_compression_capacity']))
                props += ['compression']
        if self.flash and int(data['license_flash']) != self.flash:
            props += ['flash']
        if self.cloud and int(data['license_cloud_enclosures']) != self.cloud:
            props += ['cloud']
        if self.easytier and int(data['license_easy_tier']) != self.easytier:
            props += ['easytier']
        if self.physical_flash and data['license_physical_flash'] != self.physical_flash:
            props += ['physical_flash']
        self.log('props: %s', props)
        return props

    def license_update(self, modify):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'chlicense'
        for license in modify:
            cmdopts = {}
            cmdopts[license] = getattr(self, license)
            self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.changed = True if modify else False
        if self.encryption:
            cmdopts = {}
            cmdopts['encryption'] = self.encryption
            self.changed = True
            self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        self.log('Licensed functions %s updated', modify)
        self.message += ' Licensed functions %s updated.' % modify

    def license_key_update(self):
        existing_license_keys = []
        license_id_pairs = {}
        license_add_remove = False
        if self.module.check_mode:
            self.changed = True
            return
        for key in self.license_key:
            if key == 'None':
                self.log(' Empty License key list provided')
                return
        cmd = 'lsfeature'
        cmdopts = {}
        feature_list = self.restapi.svc_run_command(cmd, cmdopts, cmdargs=None)
        for feature in feature_list:
            existing_license_keys.append(feature['license_key'])
            license_id_pairs[feature['license_key']] = feature['id']
        self.log('existing licenses=%s, license_id_pairs=%s', existing_license_keys, license_id_pairs)
        if set(existing_license_keys).symmetric_difference(set(self.license_key)):
            license_add_remove = True
        if license_add_remove:
            deactivate_license_keys, activate_license_keys = (False, False)
            deactivate_license_keys = list(set(existing_license_keys) - set(self.license_key))
            self.log('deactivate_license_keys %s ', deactivate_license_keys)
            if deactivate_license_keys:
                for item in deactivate_license_keys:
                    if not item:
                        self.log('%s item', [license_id_pairs[item]])
                        self.restapi.svc_run_command('deactivatefeature', None, [license_id_pairs[item]])
                        self.changed = True
                        self.log('%s deactivated', deactivate_license_keys)
                self.message += ' License %s deactivated.' % deactivate_license_keys
            activate_license_keys = list(set(self.license_key) - set(existing_license_keys))
            self.log('activate_license_keys %s ', activate_license_keys)
            if activate_license_keys:
                for item in activate_license_keys:
                    if item:
                        self.restapi.svc_run_command('activatefeature', {'licensekey': item}, None)
                        self.changed = True
                        self.log('%s activated', activate_license_keys)
                self.message += ' License %s activated.' % activate_license_keys
        else:
            self.message += ' No license Changes.'

    def apply(self):
        msg = None
        modify = []
        self.basic_checks()
        self.system_data = self.get_system_info()
        if self.systemname or self.ntpip or self.timezone or self.time:
            self.system_update(self.system_data)
        self.dns_configure()
        modify = self.license_probe()
        if modify:
            self.license_update(modify)
        if self.license_key:
            self.license_key_update()
        if self.changed:
            if self.module.check_mode:
                msg = 'skipping changes due to check mode.'
            else:
                msg = self.message
        else:
            msg = 'No modifications required. Exiting with no changes.'
        self.module.exit_json(msg=msg, changed=self.changed)