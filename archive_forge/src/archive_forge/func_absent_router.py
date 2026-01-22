from __future__ import absolute_import, division, print_function
from ansible.module_utils.basic import AnsibleModule
from ..module_utils.cloudstack import (AnsibleCloudStack, cs_argument_spec,
def absent_router(self):
    router = self.get_router()
    if router:
        self.result['changed'] = True
        args = {'id': router['id']}
        if not self.module.check_mode:
            res = self.query_api('destroyRouter', **args)
            poll_async = self.module.params.get('poll_async')
            if poll_async:
                self.poll_job(res, 'router')
        return router