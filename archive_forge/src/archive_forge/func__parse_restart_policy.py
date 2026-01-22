import copy
import shlex
from heat.common import exception
from heat.common.i18n import _
from heat.engine import attributes
from heat.engine.clients import progress
from heat.engine import constraints
from heat.engine import properties
from heat.engine import resource
from heat.engine.resources.openstack.nova import server_network_mixin
from heat.engine import support
from heat.engine import translation
def _parse_restart_policy(self, policy):
    restart_policy = None
    if ':' in policy:
        policy, count = policy.split(':')
        if policy in ['on-failure']:
            restart_policy = {'Name': policy, 'MaximumRetryCount': count or '0'}
    elif policy in ['always', 'unless-stopped', 'on-failure', 'no']:
        restart_policy = {'Name': policy, 'MaximumRetryCount': '0'}
    return restart_policy