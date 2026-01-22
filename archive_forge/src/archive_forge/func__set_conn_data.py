from __future__ import (absolute_import, division, print_function)
import fcntl
import os
import os.path
import subprocess
import re
from ansible.compat import selectors
from ansible.errors import AnsibleError, AnsibleFileNotFound
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils.common.process import get_bin_path
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase, BUFSIZE
from ansible.utils.display import Display
from ansible_collections.community.docker.plugins.module_utils.version import LooseVersion
def _set_conn_data(self):
    """ initialize for the connection, cannot do only in init since all data is not ready at that point """
    self._set_docker_args()
    self.remote_user = self.get_option('remote_user')
    if self.remote_user is None and self._play_context.remote_user is not None:
        self.remote_user = self._play_context.remote_user
    self.timeout = self.get_option('container_timeout')
    if self.timeout == 10 and self.timeout != self._play_context.timeout:
        self.timeout = self._play_context.timeout