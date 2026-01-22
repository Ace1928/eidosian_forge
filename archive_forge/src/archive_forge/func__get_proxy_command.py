from __future__ import absolute_import, division, print_function
import json
import logging
import os
from ansible.errors import AnsibleConnectionFailure, AnsibleError
from ansible.module_utils._text import to_bytes, to_native, to_text
from ansible.module_utils.basic import missing_required_lib
from ansible.module_utils.parsing.convert_bool import BOOLEANS_FALSE, BOOLEANS_TRUE
from ansible.module_utils.six import PY3
from ansible.module_utils.six.moves import cPickle
from ansible.playbook.play_context import PlayContext
from ansible.plugins.connection import ensure_connect
from ansible.plugins.loader import netconf_loader
from ansible_collections.ansible.netcommon.plugins.plugin_utils.connection_base import (
from ansible_collections.ansible.netcommon.plugins.plugin_utils.version import Version
def _get_proxy_command(self, port=22):
    proxy_command = None
    proxy_command = self.get_option('proxy_command')
    sock = None
    if proxy_command:
        if Version(NCCLIENT_VERSION) < '0.6.10':
            raise AnsibleError('Configuring jumphost settings through ProxyCommand is unsupported in ncclient version %s. Please upgrade to ncclient 0.6.10 or newer.' % NCCLIENT_VERSION)
        replacers = {'%h': self._play_context.remote_addr, '%p': port, '%r': self._play_context.remote_user}
        for find, replace in replacers.items():
            proxy_command = proxy_command.replace(find, str(replace))
        sock = ProxyCommand(proxy_command)
    return sock