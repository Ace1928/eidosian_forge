from __future__ import absolute_import, division, print_function
from ansible.module_utils.connection import Connection
from ansible.module_utils.six.moves.urllib.parse import quote_plus
from ansible.plugins.action import ActionBase
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common import utils
from ansible_collections.ansible.utils.plugins.module_utils.common.argspec_validate import (
from ansible_collections.splunk.es.plugins.module_utils.splunk import (
from ansible_collections.splunk.es.plugins.modules.splunk_data_inputs_monitor import DOCUMENTATION
def _check_argspec(self):
    aav = AnsibleArgSpecValidator(data=utils.remove_empties(self._task.args), schema=DOCUMENTATION, schema_format='doc', name=self._task.action)
    valid, errors, self._task.args = aav.validate()
    if not valid:
        self._result['failed'] = True
        self._result['msg'] = errors