from __future__ import absolute_import, division, print_function
from ansible_collections.google.cloud.plugins.module_utils.gcp_utils import navigate_hash, GcpSession, GcpModule, replace_resource_dict
from ansible.module_utils._text import to_native
import json
import os
import base64
Main function