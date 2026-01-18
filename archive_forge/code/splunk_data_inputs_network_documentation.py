from __future__ import absolute_import, division, print_function
from ansible.errors import AnsibleActionFail
from ansible.module_utils.connection import Connection
from ansible.module_utils.six.moves.urllib.parse import quote_plus
from ansible.plugins.action import ActionBase
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.splunk.es.plugins.module_utils.splunk import (
from ansible_collections.splunk.es.plugins.modules.splunk_data_inputs_network import DOCUMENTATION
Replace the AnsibleModule fail_json here
        :param msg: The message for the failure
        :type msg: str
        