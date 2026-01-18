from __future__ import absolute_import, division, print_function
from copy import deepcopy
import re
from ansible.module_utils._text import to_text
from ansible.module_utils.basic import AnsibleModule, env_fallback
from ansible.module_utils.common.validation import check_required_together
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import remove_default_spec
from ansible_collections.community.network.plugins.module_utils.network.icx.icx import get_config, load_config
 main entry point for module execution
    