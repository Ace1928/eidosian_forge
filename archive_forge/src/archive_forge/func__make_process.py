from __future__ import absolute_import, division, print_function
from ansible_collections.community.general.plugins.module_utils.module_helper import StateModuleHelper
from ansible_collections.community.general.plugins.module_utils.gconftool2 import gconftool2_runner
def _make_process(self, fail_on_err):

    def process(rc, out, err):
        if err and fail_on_err:
            self.ansible.fail_json(msg='gconftool-2 failed with error: %s' % str(err))
        out = out.rstrip()
        self.vars.value = None if out == '' else out
        return self.vars.value
    return process