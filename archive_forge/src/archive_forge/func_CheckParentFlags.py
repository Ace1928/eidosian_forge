from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resource_manager import exceptions
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
import six
def CheckParentFlags(args, parent_required=True):
    """Assert that there are no conflicts with parent flags.

  Ensure that both the organization flag and folder flag are not set at the
  same time. This is a little tricky since the folder flag doesn't exist for
  all commands which accept a parent specification.

  Args:
    args: The argument object
    parent_required: True to assert that a parent flag was set
  """
    if getattr(args, 'folder', None) and args.organization:
        raise calliope_exceptions.ConflictingArgumentsException('--folder', '--organization')
    if parent_required:
        if 'folder' in args and (not args.folder) and (not args.organization):
            raise exceptions.ArgumentError('Neither --folder nor --organization provided, exactly one required')
        elif 'folder' not in args and (not args.organization):
            raise exceptions.ArgumentError('--organization is required')