from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import (
import json
import re
import time
def _remove_instances_url(self):
    return 'https://www.googleapis.com/compute/v1/projects/{project}/zones/{zone}/instanceGroups/{name}/removeInstances'.format(**self.module.params)