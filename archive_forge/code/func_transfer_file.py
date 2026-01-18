from __future__ import (absolute_import, division, print_function)
import re
import os
import sys
import time
from xml.etree import ElementTree
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.community.network.plugins.module_utils.network.cloudengine.ce import ce_argument_spec, get_nc_config
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import validate_ip_v6_address
def transfer_file(self, dest):
    """Begin to transfer file by scp"""
    if not self.local_file_exists():
        self.module.fail_json(msg="Could not transfer file. Local file doesn't exist.")
    if not self.enough_space():
        self.module.fail_json(msg='Could not transfer file. Not enough space on device.')
    hostname = self.module.params['provider']['host']
    username = self.module.params['provider']['username']
    password = self.module.params['provider']['password']
    port = self.module.params['provider']['port']
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    ssh.connect(hostname=hostname, username=username, password=password, port=port)
    full_remote_path = '{0}{1}'.format(self.file_system, dest)
    scp = SCPClient(ssh.get_transport())
    try:
        scp.put(self.local_file, full_remote_path)
    except Exception:
        time.sleep(10)
        file_exists, temp_size = self.remote_file_exists(dest, self.file_system)
        file_size = os.path.getsize(self.local_file)
        if file_exists and int(temp_size) == int(file_size):
            pass
        else:
            scp.close()
            self.module.fail_json(msg='Could not transfer file. There was an error during transfer. Please make sure the format of input parameters is right.')
    scp.close()
    return True