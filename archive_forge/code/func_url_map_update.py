from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import time
def url_map_update(module, request, response):
    auth = GcpSession(module, 'compute')
    auth.post(''.join(['https://compute.googleapis.com/compute/v1/', 'projects/{project}/targetHttpProxies/{name}/setUrlMap']).format(**module.params), {u'urlMap': replace_resource_dict(module.params.get(u'url_map', {}), 'selfLink')})