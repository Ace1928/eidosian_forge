from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
def topic_pattern(name, module):
    if name is None:
        return
    regex = 'projects/.*/topics/.*'
    if not re.match(regex, name):
        formatted_params = {'project': module.params['project'], 'topic': replace_resource_dict(module.params['topic'], 'name')}
        name = 'projects/{project}/topics/{topic}'.format(**formatted_params)
    return name