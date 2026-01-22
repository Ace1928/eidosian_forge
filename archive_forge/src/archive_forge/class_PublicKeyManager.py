from __future__ import absolute_import, division, print_function
import collections
import os
from copy import deepcopy
from functools import partial
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.iosxr import (
from ansible_collections.cisco.iosxr.plugins.module_utils.network.iosxr.utils.utils import Version
class PublicKeyManager(object):

    def __init__(self, module, result):
        self._module = module
        self._result = result

    def convert_key_to_base64(self):
        """IOS-XR only accepts base64 decoded files, this converts the public key to a temp file."""
        if self._module.params['aggregate']:
            name = 'aggregate'
        else:
            name = self._module.params['name']
        if self._module.params['public_key_contents']:
            key = self._module.params['public_key_contents']
        elif self._module.params['public_key']:
            readfile = open(self._module.params['public_key'], 'r')
            key = readfile.read()
        splitfile = key.split()[1]
        base64key = b64decode(splitfile)
        base64file = open('/tmp/publickey_%s.b64' % name, 'wb')
        base64file.write(base64key)
        base64file.close()
        return '/tmp/publickey_%s.b64' % name

    def copy_key_to_node(self, base64keyfile):
        """Copy key to IOS-XR node. We use SFTP because older IOS-XR versions don't handle SCP very well."""
        if self._module.params['aggregate']:
            name = 'aggregate'
        else:
            name = self._module.params['name']
        src = base64keyfile
        dst = '/harddisk:/publickey_%s.b64' % name
        copy_file(self._module, src, dst)

    def addremovekey(self, command):
        """Add or remove key based on command"""
        admin = self._module.params.get('admin')
        conn = get_connection(self._module)
        if admin:
            conn.send_command('admin')
        out = conn.send_command(command, prompt='yes/no', answer='yes')
        if admin:
            conn.send_command('exit')
        return out

    def run(self):
        if self._module.params['state'] == 'present':
            if not self._module.check_mode:
                key = self.convert_key_to_base64()
                self.copy_key_to_node(key)
                if self._module.params['aggregate']:
                    for user in self._module.params['aggregate']:
                        cmdtodo = 'crypto key import authentication rsa username %s harddisk:/publickey_aggregate.b64' % user
                        self.addremovekey(cmdtodo)
                else:
                    cmdtodo = 'crypto key import authentication rsa username %s harddisk:/publickey_%s.b64' % (self._module.params['name'], self._module.params['name'])
                    self.addremovekey(cmdtodo)
        elif self._module.params['state'] == 'absent':
            if not self._module.check_mode:
                if self._module.params['aggregate']:
                    for user in self._module.params['aggregate']:
                        cmdtodo = 'crypto key zeroize authentication rsa username %s' % user
                        self.addremovekey(cmdtodo)
                else:
                    cmdtodo = 'crypto key zeroize authentication rsa username %s' % self._module.params['name']
                    self.addremovekey(cmdtodo)
        elif self._module.params['purge'] is True:
            if not self._module.check_mode:
                cmdtodo = 'crypto key zeroize authentication rsa all'
                self.addremovekey(cmdtodo)
        return self._result