from __future__ import absolute_import, division, print_function
import ansible_collections.community.rabbitmq.plugins.module_utils.version as Version
import json
import re
from ansible.module_utils.basic import AnsibleModule
from ansible.module_utils.common.collections import count
def _get_topic_permissions(self):
    """Get topic permissions of the user from RabbitMQ."""
    if self._version < Version.StrictVersion('3.7.0'):
        return dict()
    if self._version >= Version.StrictVersion('3.7.6'):
        permissions = json.loads(self._exec(['list_user_topic_permissions', self.username, '--formatter', 'json']))
    else:
        output = self._exec(['list_user_topic_permissions', self.username]).strip().split('\n')
        perms_out = [perm.split('\t') for perm in output if perm.strip()]
        permissions = list()
        for vhost, exchange, write, read in perms_out:
            permissions.append(dict(vhost=vhost, exchange=exchange, write=write, read=read))
    return as_topic_permission_dict(permissions)