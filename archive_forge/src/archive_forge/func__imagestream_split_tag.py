from __future__ import (absolute_import, division, print_function)
import traceback
import copy
from ansible.module_utils._text import to_native
from ansible.module_utils.parsing.convert_bool import boolean
from ansible.module_utils.six import string_types
from ansible_collections.community.okd.plugins.module_utils.openshift_common import AnsibleOpenshiftModule
from ansible_collections.community.okd.plugins.module_utils.openshift_docker_image import (
def _imagestream_split_tag(name):
    parts = name.split(':')
    name = parts[0]
    tag = ''
    if len(parts) > 1:
        tag = parts[1]
    if len(tag) == 0:
        tag = 'latest'
    return (name, tag, len(parts) == 2)