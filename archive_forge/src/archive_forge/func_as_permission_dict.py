from __future__ import absolute_import, division, print_function
import ansible_collections.community.rabbitmq.plugins.module_utils.version as Version
import json
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.collections import count
def as_permission_dict(vhost_permission_list):
    return dict([(vhost_permission['vhost'], vhost_permission) for vhost_permission in normalized_permissions(vhost_permission_list)])