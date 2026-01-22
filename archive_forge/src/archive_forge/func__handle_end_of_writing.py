from __future__ import (absolute_import, division, print_function)
import os
import os.path
import socket as pysocket
import struct
from ansible.module_utils.six import PY2
from ansible_collections.community.docker.plugins.module_utils._api.utils import socket as docker_socket
from ansible_collections.community.docker.plugins.module_utils.socket_helper import (
def _handle_end_of_writing(self):
    if self._end_of_writing and len(self._write_buffer) == 0:
        self._end_of_writing = False
        self._log('Shutting socket down for writing')
        shutdown_writing(self._sock, self._log)