from __future__ import absolute_import, division, print_function
import json
import re
from copy import deepcopy
from ansible.module_utils._text import to_text
from ansible.module_utils.common._collections_compat import Mapping
from ansible.module_utils.connection import Connection, ConnectionError
from ansible.module_utils.six import PY2, PY3
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.config import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
def to_command(module, commands):
    transform = ComplexList(dict(command=dict(key=True), output=dict(type='str', default='text'), prompt=dict(type='list'), answer=dict(type='list'), newline=dict(type='bool', default=True), sendonly=dict(type='bool', default=False), check_all=dict(type='bool', default=False)), module)
    commands = transform(to_list(commands))
    for item in commands:
        if is_json(item['command']):
            item['output'] = 'json'
    return commands