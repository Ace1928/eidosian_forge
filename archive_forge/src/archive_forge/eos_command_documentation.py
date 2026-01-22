from __future__ import absolute_import, division, print_function
import time
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.parsing import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import to_lines
from ansible_collections.arista.eos.plugins.module_utils.network.eos.eos import (
entry point for module execution