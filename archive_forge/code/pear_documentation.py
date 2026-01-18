from __future__ import absolute_import, division, print_function
import os
from ansible.module_utils.common.text.converters import to_text
from ansible.module_utils.basic import AnsibleModule
Query the package status in both the local system and the repository.
    Returns a boolean to indicate if the package is installed,
    and a second boolean to indicate if the package is up-to-date.