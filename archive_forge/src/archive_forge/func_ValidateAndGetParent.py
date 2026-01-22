from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import re
from googlecloudsdk.command_lib.scc import errors
from googlecloudsdk.command_lib.scc import util as scc_util
def ValidateAndGetParent(args):
    """Validates parent."""
    if args.organization is not None:
        name_pattern = re.compile('^organizations/[0-9]{1,19}$')
        id_pattern = re.compile('^[0-9]{1,19}$')
        if name_pattern.match(args.organization):
            return args.organization
        if id_pattern.match(args.organization):
            return f'organizations/{args.organization}'
        if '/' in args.organization:
            raise errors.InvalidSCCInputError("When providing a full resource path, it must include the pattern '^organizations/[0-9]{1,19}$'.")
        raise errors.InvalidSCCInputError("Organization does not match the pattern '^[0-9]{1,19}$'.")
    if args.folder is not None:
        if '/' in args.folder:
            pattern = re.compile('^folders/.*$')
            if not pattern.match(args.folder):
                raise errors.InvalidSCCInputError("When providing a full resource path, it must include the pattern '^folders/.*$'.")
            else:
                return args.folder
        else:
            return f'folders/{args.folder}'
    if args.project is not None:
        if '/' in args.project:
            pattern = re.compile('^projects/.*$')
            if not pattern.match(args.project):
                raise errors.InvalidSCCInputError("When providing a full resource path, it must include the pattern '^projects/.*$'.")
            else:
                return args.project
        else:
            return f'projects/{args.project}'