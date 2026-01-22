from __future__ import absolute_import, division, print_function
import json
from ansible.module_utils.urls import fetch_url, url_argument_spec
from ansible.module_utils.common.text.converters import to_native
def api_argument_spec():
    """
    Creates an argument spec that can be used with any module
    that will be requesting content via Rundeck API
    """
    api_argument_spec = url_argument_spec()
    api_argument_spec.update(dict(url=dict(required=True, type='str'), api_version=dict(type='int', default=39), api_token=dict(required=True, type='str', no_log=True)))
    return api_argument_spec