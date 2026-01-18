from __future__ import absolute_import, division, print_function
import json
from ansible.plugins.action import ActionBase
from ansible.module_utils.connection import Connection
from ansible.module_utils._text import to_text
from ansible.module_utils.six.moves.urllib.parse import quote
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import (
from ansible_collections.ibm.qradar.plugins.module_utils.qradar import (
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.ibm.qradar.plugins.modules.qradar_analytics_rules import (
The fn TC of MERGE operation
        :param qradar_request: Qradar connection request
        :param module_config_params: Module input config
        :rtype: A dict
        :returns: Merge output with before and after dict
        