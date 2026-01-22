from __future__ import absolute_import, division, print_function
import time
import traceback
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_bytes
def _absent(self, znode):
    if self.exists(znode):
        self.zk.delete(znode, recursive=self.module.params['recursive'])
        return (True, {'changed': True, 'msg': 'The znode was deleted.'})
    else:
        return (True, {'changed': False, 'msg': 'The znode does not exist.'})