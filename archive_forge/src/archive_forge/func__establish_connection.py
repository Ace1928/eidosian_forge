from __future__ import absolute_import, division, print_function
import re
from os.path import exists, getsize
from socket import gaierror
from ssl import SSLError
from time import sleep
import traceback
from ansible.errors import AnsibleError, AnsibleFileNotFound, AnsibleConnectionFailure
from ansible.module_utils._text import to_bytes, to_native
from ansible.plugins.connection import ConnectionBase
from ansible.module_utils.basic import missing_required_lib
def _establish_connection(self):
    connection_kwargs = {'host': self.vmware_host, 'user': self.get_option('vmware_user'), 'pwd': self.get_option('vmware_password'), 'port': self.get_option('vmware_port')}
    if not self.validate_certs:
        if HAS_URLLIB3:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
        connection_kwargs['disableSslCertValidation'] = True
    try:
        self._si = SmartConnect(**connection_kwargs)
    except SSLError:
        raise AnsibleError('SSL Error: Certificate verification failed.')
    except gaierror:
        raise AnsibleError("Connection Error: Unable to connect to '%s'." % to_native(connection_kwargs['host']))
    except vim.fault.InvalidLogin as e:
        raise AnsibleError('Connection Login Error: %s' % to_native(e.msg))