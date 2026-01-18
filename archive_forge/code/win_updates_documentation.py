from __future__ import (absolute_import, division, print_function)
import traceback
from ansible.errors import AnsibleActionFail, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_native, to_text
from ansible.module_utils.common.validation import check_type_bool, check_type_int
from ansible.plugins.action import ActionBase
from ansible.utils.display import Display
from ..plugin_utils._reboot import reboot_host
 Process the progress result and store it in a structured object. 