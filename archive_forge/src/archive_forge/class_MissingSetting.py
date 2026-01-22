from __future__ import (absolute_import, division, print_function)
import ansible.plugins.loader as plugin_loader
from ansible import constants as C
from ansible.errors import AnsibleError, AnsibleLookupError, AnsibleOptionsError
from ansible.module_utils.common.text.converters import to_native
from ansible.module_utils.six import string_types
from ansible.plugins.lookup import LookupBase
from ansible.utils.sentinel import Sentinel
class MissingSetting(AnsibleOptionsError):
    pass