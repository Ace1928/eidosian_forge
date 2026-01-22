from __future__ import (absolute_import, division, print_function)
import json
from ansible.module_utils._text import to_bytes, to_native
from ansible_collections.containers.podman.plugins.module_utils.podman.common import LooseVersion
from ansible_collections.containers.podman.plugins.module_utils.podman.common import lower_keys
from ansible_collections.containers.podman.plugins.module_utils.podman.common import generate_systemd
from ansible_collections.containers.podman.plugins.module_utils.podman.common import delete_systemd
class PodmanPodDefaults:

    def __init__(self, module, podman_version):
        self.module = module
        self.version = podman_version
        self.defaults = {'add_host': [], 'dns': [], 'dns_opt': [], 'dns_search': [], 'infra': True, 'label': {}}

    def default_dict(self):
        return self.defaults