from __future__ import (absolute_import, division, print_function)
import os
import os.path
from ansible.errors import AnsibleFileNotFound, AnsibleConnectionFailure
from ansible.module_utils.common.text.converters import to_bytes, to_native, to_text
from ansible.plugins.connection import ConnectionBase
from ansible.utils.display import Display
from ansible_collections.community.docker.plugins.module_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils.copy import (
from ansible_collections.community.docker.plugins.plugin_utils.socket_handler import (
from ansible_collections.community.docker.plugins.plugin_utils.common_api import (
from ansible_collections.community.docker.plugins.module_utils._api.errors import APIError, DockerException, NotFound
def append_become_output(stream_id, data):
    become_output[0] += data