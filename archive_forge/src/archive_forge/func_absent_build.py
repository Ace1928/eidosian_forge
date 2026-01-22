from __future__ import absolute_import, division, print_function
import traceback
from time import sleep
from ansible.module_utils.basic import AnsibleModule, missing_required_lib
from ansible.module_utils.common.text.converters import to_native
def absent_build(self):
    try:
        self.server.delete_build(self.name, self.build_number)
    except Exception as e:
        self.module.fail_json(msg='Unable to delete build for %s: %s' % (self.jenkins_url, to_native(e)), exception=traceback.format_exc())