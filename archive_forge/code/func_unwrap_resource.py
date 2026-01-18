from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import time
def unwrap_resource(result, module):
    query_predicate = unwrap_resource_filter(module)
    matched_items = []
    for item in result:
        if all((item[k] == query_predicate[k] for k in query_predicate.keys())):
            matched_items.append(item)
    if len(matched_items) > 1:
        module.fail_json(msg='More than 1 result found: %s' % matched_items)
    if matched_items:
        return matched_items[0]
    else:
        return None