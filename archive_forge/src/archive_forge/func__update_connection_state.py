from __future__ import (annotations, absolute_import, division, print_function)
import collections.abc as c
import fcntl
import io
import os
import shlex
import typing as t
from abc import abstractmethod
from functools import wraps
from ansible import constants as C
from ansible.module_utils.common.text.converters import to_bytes, to_text
from ansible.playbook.play_context import PlayContext
from ansible.plugins import AnsiblePlugin
from ansible.plugins.become import BecomeBase
from ansible.plugins.shell import ShellBase
from ansible.utils.display import Display
from ansible.plugins.loader import connection_loader, get_shell_plugin
from ansible.utils.path import unfrackpath
def _update_connection_state(self) -> None:
    """
        Reconstruct the connection socket_path and check if it exists

        If the socket path exists then the connection is active and set
        both the _socket_path value to the path and the _connected value
        to True.  If the socket path doesn't exist, leave the socket path
        value to None and the _connected value to False
        """
    ssh = connection_loader.get('ssh', class_only=True)
    control_path = ssh._create_control_path(self._play_context.remote_addr, self._play_context.port, self._play_context.remote_user, self._play_context.connection, self._ansible_playbook_pid)
    tmp_path = unfrackpath(C.PERSISTENT_CONTROL_PATH_DIR)
    socket_path = unfrackpath(control_path % dict(directory=tmp_path))
    if os.path.exists(socket_path):
        self._connected = True
        self._socket_path = socket_path