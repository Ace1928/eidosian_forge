from __future__ import (absolute_import, division, print_function)
import os
from ansible.module_utils.basic import AnsibleModule
import json
from ansible_collections.containers.podman.plugins.module_utils.podman.common import compare_systemd_file_content
Main function of this script.