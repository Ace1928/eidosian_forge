from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.core import properties
def GetParentFromPositionalArguments(args):
    """Converts user input to one of: organization, project, or folder."""
    id_pattern = re.compile('[0-9]+')
    parent = None
    if hasattr(args, 'parent'):
        if not args.parent:
            parent = properties.VALUES.scc.parent.Get()
        else:
            parent = args.parent
    if parent is None:
        parent = properties.VALUES.scc.organization.Get()
    if parent is None and hasattr(args, 'organization'):
        parent = args.organization
    if parent is None:
        raise errors.InvalidSCCInputError('Could not find Parent argument. Please provide the parent argument.')
    if id_pattern.match(parent):
        parent = 'organizations/' + parent
    if not (parent.startswith('organizations/') or parent.startswith('projects/') or parent.startswith('folders/')):
        error_message = 'Parent must match either [0-9]+, organizations/[0-9]+, projects/.* or folders/.*.'
        raise errors.InvalidSCCInputError(error_message)
    return parent