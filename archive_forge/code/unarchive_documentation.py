from __future__ import (absolute_import, division, print_function)
import os
from ansible.errors import AnsibleError, AnsibleAction, AnsibleActionFail, AnsibleActionSkip
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.plugins.action import ActionBase
 handler for unarchive operations 