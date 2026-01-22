from __future__ import absolute_import, division, print_function
from ast import literal_eval
from ansible.module_utils._text import to_text
from ansible.module_utils.common.validation import check_required_arguments
from ansible.module_utils.connection import ConnectionError
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.cfg.base import (
from ansible_collections.ansible.netcommon.plugins.module_utils.network.common.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.facts.facts import Facts
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.utils import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.sonic import (
from ansible_collections.dellemc.enterprise_sonic.plugins.module_utils.network.sonic.utils.formatted_diff_utils import (
def _handle_failure_response(self, connection_error):
    log = None
    try:
        response = literal_eval(connection_error.args[0])
        error_app_tag = response['ietf-restconf:errors']['error'][0].get('error-app-tag')
    except Exception:
        pass
    else:
        if error_app_tag == 'too-many-elements':
            log = 'Exceeds maximum number of ACL / ACL Rules'
        elif error_app_tag == 'update-not-allowed':
            log = 'Creating ACLs with same name and different type not allowed'
    if log:
        response.update({u'log': log})
        self._module.fail_json(msg=to_text(response), code=connection_error.code)
    else:
        self._module.fail_json(msg=str(connection_error), code=connection_error.code)