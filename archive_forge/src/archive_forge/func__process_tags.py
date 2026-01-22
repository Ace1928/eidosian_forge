from __future__ import absolute_import, division, print_function
import os
import sys
import time
import traceback
from ansible.module_utils._text import to_text, to_native
from ansible.module_utils.basic import missing_required_lib, env_fallback
def _process_tags(self, resource, resource_type, tags, operation='create'):
    if tags:
        self.result['changed'] = True
        if not self.module.check_mode:
            args = {'resourceids': resource['id'], 'resourcetype': resource_type, 'tags': tags}
            if operation == 'create':
                response = self.query_api('createTags', **args)
            else:
                response = self.query_api('deleteTags', **args)
            self.poll_job(response)