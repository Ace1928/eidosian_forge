from __future__ import absolute_import, division, print_function
import os
import json
from ansible.module_utils.basic import AnsibleModule
def _process_yarn_error(self, err):
    try:
        for line in err.splitlines():
            if json.loads(line)['type'] == 'error':
                self.module.fail_json(msg=err)
    except Exception:
        self.module.fail_json(msg='Unexpected stderr output from Yarn: %s' % err, stderr=err)