from __future__ import absolute_import, division, print_function
from traceback import format_exc
import json
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_utils import (
from ansible_collections.ibm.spectrum_virtualize.plugins.module_utils.ibm_svc_ssh import IBMSVCssh
from ansible.module_utils._text import to_native
class IBMSVTrustStore:

    def __init__(self):
        argument_spec = svc_ssh_argument_spec()
        argument_spec.update(dict(password=dict(type='str', required=False, no_log=True), name=dict(type='str'), usesshkey=dict(type='str', default='no', choices=['yes', 'no']), key_filename=dict(type='str'), state=dict(type='str', choices=['present', 'absent'], required=True), remote_clustername=dict(type='str', required=True), remote_username=dict(type='str'), remote_password=dict(type='str', no_log=True)))
        self.module = AnsibleModule(argument_spec=argument_spec, supports_check_mode=True)
        self.log_path = self.module.params['log_path']
        log = get_logger(self.__class__.__name__, self.log_path)
        self.log = log.info
        self.state = self.module.params['state']
        self.remote_clustername = self.module.params['remote_clustername']
        self.usesshkey = self.module.params['usesshkey']
        self.key_filename = self.module.params['key_filename']
        self.password = self.module.params.get('password', '')
        self.name = self.module.params.get('name', '')
        self.remote_username = self.module.params.get('remote_username', '')
        self.remote_password = self.module.params.get('remote_password', '')
        if not self.name:
            self.name = 'store_{0}'.format(self.remote_clustername)
        if not self.password:
            if self.usesshkey == 'yes':
                self.log('password is none and use ssh private key. Check for its path')
                if self.key_filename:
                    self.log('key file_name is provided, use it')
                    self.look_for_keys = True
                else:
                    self.log('key file_name is not provided, use default one, ~/.ssh/id_rsa.pub')
                    self.look_for_keys = True
            else:
                self.log('password is none and SSH key is not provided')
                self.module.fail_json(msg='You must pass either password or usesshkey parameter.')
        else:
            self.log('password is given')
            self.look_for_keys = False
        self.basic_checks()
        self.changed = False
        self.msg = ''
        self.ssh_client = IBMSVCssh(module=self.module, clustername=self.module.params['clustername'], username=self.module.params['username'], password=self.password, look_for_keys=self.look_for_keys, key_filename=self.key_filename, log_path=self.log_path)

    def basic_checks(self):
        if self.state == 'present':
            if not self.remote_clustername:
                self.module.fail_json(msg='Missing mandatory parameter: remote_clustername')
            if not self.remote_username:
                self.module.fail_json(msg='Missing mandatory parameter: remote_username')
            if not self.remote_password:
                self.module.fail_json(msg='Missing mandatory parameter: remote_password')
        elif self.state == 'absent':
            if not self.remote_clustername:
                self.module.fail_json(msg='Missing mandatory parameter: remote_clustername')
            unsupported = ('remote_username', 'remote_password')
            unsupported_exists = ', '.join((field for field in unsupported if getattr(self, field)))
            if unsupported_exists:
                self.module.fail_json(msg='state=absent but following paramters have been passed: {0}'.format(unsupported_exists))

    def raise_error(self, stderr):
        message = stderr.read().decode('utf-8')
        if len(message) > 0:
            self.log('%s', message)
            self.module.fail_json(msg=message)
        else:
            message = 'Unknown error received.'
            self.module.fail_json(msg=message)

    def is_truststore_exists(self):
        merged_result = {}
        cmd = 'lstruststore -json {0}'.format(self.name)
        stdin, stdout, stderr = self.ssh_client.client.exec_command(cmd)
        result = stdout.read().decode('utf-8')
        if result:
            result = json.loads(result)
        else:
            return merged_result
        rc = stdout.channel.recv_exit_status()
        if rc > 0:
            message = stderr.read().decode('utf-8')
            if message.count('CMMVC5804E') != 1 or message.count('CMMVC6035E') != 1:
                self.log('Error in executing CLI command: %s', cmd)
                self.log('%s', message)
                self.module.fail_json(msg=message)
            else:
                self.log('Expected error: %s', message)
        if isinstance(result, list):
            for d in result:
                merged_result.update(d)
        else:
            merged_result = result
        return merged_result

    def download_file(self):
        if self.module.check_mode:
            return
        cmd = 'scp -o stricthostkeychecking=no {0}@{1}:/dumps/certificate.pem /upgrade/'.format(self.remote_username, self.remote_clustername)
        self.log('Command to be executed: %s', cmd)
        stdin, stdout, stderr = self.ssh_client.client.exec_command(cmd, get_pty=True, timeout=60 * 1.5)
        result = ''
        while not stdout.channel.recv_ready():
            data = stdout.channel.recv(1024)
            self.log(str(data, 'utf-8'))
            if data:
                if b'Password:' in data or b'password' in data:
                    stdin.write('{0}\n'.format(self.remote_password))
                    stdin.flush()
                else:
                    result += data.decode('utf-8')
                break
        result += stdout.read().decode('utf-8')
        rc = stdout.channel.recv_exit_status()
        if rc > 0:
            message = stderr.read().decode('utf-8')
            self.log('Error in executing command: %s', cmd)
            if not len(message) > 1:
                if len(result) > 1:
                    err = result.replace('\rPassword:\r\n', '')
                    self.log('Error: %s', err)
                    if err:
                        self.module.fail_json(msg=err)
                self.module.fail_json(msg='Unknown error received')
            else:
                self.module.fail_json(msg=message)
        else:
            self.log(result)

    def create_truststore(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'mktruststore -name {0} -file {1}'.format(self.name, '/upgrade/certificate.pem')
        self.log('Command to be executed: %s', cmd)
        stdin, stdout, stderr = self.ssh_client.client.exec_command(cmd)
        result = stdout.read().decode('utf-8')
        rc = stdout.channel.recv_exit_status()
        if rc > 0:
            self.log('Error in executing command: %s', cmd)
            self.raise_error(stderr)
        else:
            self.log('Truststore (%s) created', self.name)
            self.log(result)
            self.changed = True

    def delete_truststore(self):
        if self.module.check_mode:
            self.changed = True
            return
        cmd = 'rmtruststore {0}'.format(self.name)
        self.log('Command to be executed: %s', cmd)
        stdin, stdout, stderr = self.ssh_client.client.exec_command(cmd)
        result = stdout.read().decode('utf-8')
        rc = stdout.channel.recv_exit_status()
        if rc > 0:
            self.log('Error in executing command: %s', cmd)
            self.raise_error(stderr)
        else:
            self.log('Truststore (%s) deleted', self.name)
            self.log(result)
            self.changed = True

    def apply(self):
        if self.is_truststore_exists():
            self.log('Truststore (%s) exists', self.name)
            if self.state == 'present':
                self.msg = 'Truststore ({0}) already exist. No modifications done'.format(self.name)
            else:
                self.delete_truststore()
                self.msg = 'Truststore ({0}) deleted.'.format(self.name)
        elif self.state == 'absent':
            self.msg = 'Truststore ({0}) does not exist. No modifications done.'.format(self.name)
        else:
            self.download_file()
            self.create_truststore()
            self.msg = 'Truststore ({0}) created.'.format(self.name)
        if self.module.check_mode:
            self.msg = 'skipping changes due to check mode.'
        self.module.exit_json(changed=self.changed, msg=self.msg)