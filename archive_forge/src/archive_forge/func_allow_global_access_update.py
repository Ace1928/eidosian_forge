from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import time
def allow_global_access_update(module, request, response):
    auth = GcpSession(module, 'compute')
    auth.patch(''.join(['https://compute.googleapis.com/compute/v1/', 'projects/{project}/regions/{region}/forwardingRules/{name}']).format(**module.params), {u'allowGlobalAccess': module.params.get('allow_global_access')})