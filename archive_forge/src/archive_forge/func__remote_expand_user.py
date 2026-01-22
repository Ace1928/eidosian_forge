from __future__ import (absolute_import, division, print_function)
import base64
import json
import os
import random
import re
import shlex
import stat
import tempfile
from abc import ABC, abstractmethod
from collections.abc import Sequence
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleConnectionFailure, AnsibleActionSkip, AnsibleActionFail, AnsibleAuthenticationFailure
from ansible.executor.module_common import modify_module
from ansible.executor.interpreter_discovery import discover_interpreter, InterpreterDiscoveryRequiredError
from ansible.module_utils.common.arg_spec import ArgumentSpecValidator
from ansible.module_utils.errors import UnsupportedError
from ansible.module_utils.json_utils import _filter_non_json_lines
from ansible.module_utils.six import binary_type, string_types, text_type
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.parsing.utils.jsonify import jsonify
from ansible.release import __version__
from ansible.utils.collection_loader import resource_from_fqcr
from ansible.utils.display import Display
from ansible.utils.unsafe_proxy import wrap_var, AnsibleUnsafeText
from ansible.vars.clean import remove_internal_keys
from ansible.utils.plugin_docs import get_versioned_doclink
def _remote_expand_user(self, path, sudoable=True, pathsep=None):
    """ takes a remote path and performs tilde/$HOME expansion on the remote host """
    if not path.startswith('~'):
        return path
    split_path = path.split(os.path.sep, 1)
    expand_path = split_path[0]
    if expand_path == '~':
        become_user = self.get_become_option('become_user')
        if getattr(self._connection, '_remote_is_local', False):
            pass
        elif sudoable and self._connection.become and become_user:
            expand_path = '~%s' % become_user
        else:
            expand_path = '~%s' % (self._get_remote_user() or '')
    cmd = self._connection._shell.expand_user(expand_path)
    data = self._low_level_execute_command(cmd, sudoable=False)
    try:
        initial_fragment = data['stdout'].strip().splitlines()[-1]
    except IndexError:
        initial_fragment = None
    if not initial_fragment:
        cmd = self._connection._shell.pwd()
        pwd = self._low_level_execute_command(cmd, sudoable=False).get('stdout', '').strip()
        if pwd:
            expanded = pwd
        else:
            expanded = path
    elif len(split_path) > 1:
        expanded = self._connection._shell.join_path(initial_fragment, *split_path[1:])
    else:
        expanded = initial_fragment
    if '..' in os.path.dirname(expanded).split('/'):
        raise AnsibleError("'%s' returned an invalid relative home directory path containing '..'" % self._get_remote_addr({}))
    return expanded