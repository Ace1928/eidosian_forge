from __future__ import absolute_import, division, print_function
import os
from ansible_collections.community.vmware.plugins.module_utils.vmware import vmware_argument_spec, get_all_objs, wait_for_task, PyVmomi
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.urls import open_url
from ansible.module_utils.six.moves.urllib.error import HTTPError
from ansible.module_utils._text import to_native
class VMwareConfigurationBackup(PyVmomi):

    def __init__(self, module):
        super(VMwareConfigurationBackup, self).__init__(module)
        self.state = self.module.params['state']
        self.dest = self.module.params['dest']
        self.src = self.module.params['src']
        self.hostname = self.module.params['hostname']
        self.username = self.module.params['username']
        self.password = self.module.params['password']
        self.validate_certs = self.module.params['validate_certs']
        self.esxi_hostname = self.module.params.get('esxi_hostname', None)
        self.host = self.find_host_system()
        self.cfg_hurl = self.hostname if self.content.about.apiType == 'HostAgent' else self.host.name

    def find_host_system(self):
        if self.esxi_hostname:
            host_system_obj = self.find_hostsystem_by_name(host_name=self.esxi_hostname)
            if host_system_obj:
                return host_system_obj
            else:
                self.module.fail_json(msg='Failed to find ESXi %s' % self.esxi_hostname)
        host_system = get_all_objs(self.content, [vim.HostSystem])
        return list(host_system)[0]

    def process_state(self):
        if self.state == 'saved':
            self.save_configuration()
        if self.state == 'absent':
            self.reset_configuration()
        if self.state == 'loaded':
            self.load_configuration()

    def load_configuration(self):
        if not os.path.isfile(self.src):
            self.module.fail_json(msg='Source file {0} does not exist'.format(self.src))
        url = self.host.configManager.firmwareSystem.QueryFirmwareConfigUploadURL()
        url = url.replace('*', self.cfg_hurl)
        if self.module.params['port'] == 443:
            url = url.replace('http:', 'https:')
        try:
            open_url(url=url, method='HEAD', validate_certs=self.validate_certs)
        except HTTPError as e:
            url = e.geturl()
        try:
            with open(self.src, 'rb') as file:
                data = file.read()
            open_url(url=url, data=data, method='PUT', validate_certs=self.validate_certs, url_username=self.username, url_password=self.password, force_basic_auth=True)
        except Exception as e:
            self.module.fail_json(msg=to_native(e))
        if not self.host.runtime.inMaintenanceMode:
            self.enter_maintenance()
        try:
            self.host.configManager.firmwareSystem.RestoreFirmwareConfiguration(force=True)
            self.module.exit_json(changed=True)
        except Exception as e:
            self.exit_maintenance()
            self.module.fail_json(msg=to_native(e))

    def reset_configuration(self):
        if not self.host.runtime.inMaintenanceMode:
            self.enter_maintenance()
        try:
            self.host.configManager.firmwareSystem.ResetFirmwareToFactoryDefaults()
            self.module.exit_json(changed=True)
        except Exception as e:
            self.exit_maintenance()
            self.module.fail_json(msg=to_native(e))

    def save_configuration(self):
        url = self.host.configManager.firmwareSystem.BackupFirmwareConfiguration()
        url = url.replace('*', self.cfg_hurl)
        if self.module.params['port'] == 443:
            url = url.replace('http:', 'https:')
        if os.path.isdir(self.dest):
            filename = url.rsplit('/', 1)[1]
            self.dest = os.path.join(self.dest, filename)
        else:
            filename, file_extension = os.path.splitext(self.dest)
            if file_extension != '.tgz':
                self.dest = filename + '.tgz'
        try:
            request = open_url(url=url, validate_certs=self.validate_certs)
            with open(self.dest, 'wb') as file:
                file.write(request.read())
            self.module.exit_json(changed=True, dest_file=self.dest)
        except IOError as e:
            error_msg = 'Failed to save %s ' % url
            error_msg += 'to %s. Ensure that the dest path exists and is writable. ' % self.dest
            error_msg += 'Details: %s' % to_native(e)
            self.module.fail_json(msg=error_msg)
        except Exception as e:
            self.module.fail_json(msg=to_native(e))

    def enter_maintenance(self):
        try:
            task = self.host.EnterMaintenanceMode_Task(timeout=15)
            success, result = wait_for_task(task)
        except Exception as e:
            self.module.fail_json(msg='Failed to enter maintenance mode. Ensure that there are no powered on machines on the host. %s' % to_native(e))

    def exit_maintenance(self):
        try:
            task = self.host.ExitMaintenanceMode_Task(timeout=15)
            success, result = wait_for_task(task)
        except Exception as generic_exc:
            self.module.fail_json(msg='Failed to exit maintenance mode due to %s' % to_native(generic_exc))