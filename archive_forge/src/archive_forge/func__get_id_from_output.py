from __future__ import absolute_import, division, print_function
import json
import re
import shlex
from ansible.module_utils._text import to_native
from ansible.module_utils.basic import AnsibleModule
from ansible_collections.containers.podman.plugins.module_utils.podman.common import run_podman_command
def _get_id_from_output(self, lines, startswith=None, contains=None, split_on=' ', maxsplit=1):
    layer_ids = []
    for line in lines.splitlines():
        if startswith and line.startswith(startswith) or (contains and contains in line):
            splitline = line.rsplit(split_on, maxsplit)
            layer_ids.append(splitline[1])
    if not layer_ids:
        layer_ids = lines.splitlines()
    return layer_ids[-1]