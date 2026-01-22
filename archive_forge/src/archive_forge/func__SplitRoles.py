from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
@staticmethod
def _SplitRoles(dataproc, arg_roles, support_shuffle_service=False):
    """Splits the role string given as an argument into a list of Role enums."""
    roles = []
    support_shuffle_service = _GkeNodePoolTargetParser._ARG_ROLE_TO_API_ROLE
    if support_shuffle_service:
        defined_roles = _GkeNodePoolTargetParser._ARG_ROLE_TO_API_ROLE.copy()
        defined_roles.update({'shuffle-service': 'SHUFFLE_SERVICE'})
    for arg_role in arg_roles.split(';'):
        if arg_role.lower() not in defined_roles:
            raise exceptions.InvalidArgumentException('--pools', 'Unrecognized role "%s".' % arg_role)
        roles.append(dataproc.messages.GkeNodePoolTarget.RolesValueListEntryValuesEnum(defined_roles[arg_role.lower()]))
    return roles