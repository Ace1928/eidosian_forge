from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, GcpRequest, replace_resource_dict
import json
import copy
import datetime
import time
def add_deletions(result, original_soa, original_record):
    if original_soa:
        result['deletions'].append(original_soa)
    if original_record:
        result['deletions'].append(original_record)