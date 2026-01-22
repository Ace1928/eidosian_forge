from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.core import properties
def GetParentFromNamedArguments(args):
    """Gets and validates parent from named arguments."""
    if args.organization is not None:
        if '/' in args.organization:
            pattern = re.compile('^organizations/[0-9]{1,19}$')
            if not pattern.match(args.organization):
                raise errors.InvalidSCCInputError("When providing a full resource path, it must include the pattern '^organizations/[0-9]{1,19}$'.")
            else:
                return args.organization
        else:
            pattern = re.compile('^[0-9]{1,19}$')
            if not pattern.match(args.organization):
                raise errors.InvalidSCCInputError("Organization does not match the pattern '^[0-9]{1,19}$'.")
            else:
                return 'organizations/' + args.organization
    if hasattr(args, 'folder') and args.folder is not None:
        if '/' in args.folder:
            pattern = re.compile('^folders/.*$')
            if not pattern.match(args.folder):
                raise errors.InvalidSCCInputError("When providing a full resource path, it must include the pattern '^folders/.*$'.")
            else:
                return args.folder
        else:
            return 'folders/' + args.folder
    if hasattr(args, 'project') and args.project is not None:
        if '/' in args.project:
            pattern = re.compile('^projects/.*$')
            if not pattern.match(args.project):
                raise errors.InvalidSCCInputError("When providing a full resource path, it must include the pattern '^projects/.*$'.")
            else:
                return args.project
        else:
            return 'projects/' + args.project