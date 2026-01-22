from __future__ import (absolute_import, division, print_function)
import os.path
from ansible import constants as C
from ansible.module_utils.six import string_types
from ansible.module_utils.six.moves import shlex_quote
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import MutableSequence
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
from ansible.plugins.loader import connection_loader
def _process_remote(self, task_args, host, path, user, port_matches_localhost_port):
    """
        :arg host: hostname for the path
        :arg path: file path
        :arg user: username for the transfer
        :arg port_matches_localhost_port: boolean whether the remote port
            matches the port used by localhost's sshd.  This is used in
            conjunction with seeing whether the host is localhost to know
            if we need to have the module substitute the pathname or if it
            is a different host (for instance, an ssh tunnelled port or an
            alternative ssh port to a vagrant host.)
        """
    transport = self._connection.transport
    if host not in C.LOCALHOST or transport != 'local' or (host in C.LOCALHOST and (not port_matches_localhost_port)):
        if port_matches_localhost_port and host in C.LOCALHOST:
            task_args['_substitute_controller'] = True
        return self._format_rsync_rsh_target(host, path, user)
    path = self._get_absolute_path(path=path)
    return path