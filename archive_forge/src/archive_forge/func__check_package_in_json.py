from __future__ import absolute_import, division, print_function
import json
import os.path
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.six import iteritems, string_types
def _check_package_in_json(json_output, package_type):
    return bool(json_output.get(package_type, []) and json_output[package_type][0].get('installed'))