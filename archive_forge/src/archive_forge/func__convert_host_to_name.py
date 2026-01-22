from __future__ import (absolute_import, division, print_function)
import datetime
import json
import copy
from functools import partial
from ansible.inventory.host import Host
from ansible.module_utils._text import to_text
from ansible.parsing.ajson import AnsibleJSONEncoder
from ansible.plugins.callback import CallbackBase
def _convert_host_to_name(self, key):
    if isinstance(key, (Host,)):
        return key.get_name()
    return key