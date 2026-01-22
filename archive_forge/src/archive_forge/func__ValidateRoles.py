from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import re
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions
@staticmethod
def _ValidateRoles(dataproc, pools):
    """Validates that roles are exclusive and that one pool has DEFAULT."""
    if not pools:
        return
    seen_roles = set()
    for pool in pools:
        for role in pool.roles:
            if role in seen_roles:
                raise exceptions.InvalidArgumentException('--pools', 'Multiple pools contained the same role "%s".' % role)
            else:
                seen_roles.add(role)
    default = dataproc.messages.GkeNodePoolTarget.RolesValueListEntryValuesEnum('DEFAULT')
    if default not in seen_roles:
        raise exceptions.InvalidArgumentException('--pools', 'If any pools are specified, then exactly one must have the "default" role.')