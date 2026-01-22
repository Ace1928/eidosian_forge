from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope.concepts import concepts
from googlecloudsdk.command_lib.util.concepts import concept_parsers
from googlecloudsdk.command_lib.util.concepts import presentation_specs
def CreateNotificationChannelResourceArg(arg_name, extra_help, required=True, plural=False):
    """Create a resource argument for a Cloud Monitoring Notification Channel.

  Args:
    arg_name: str, the name for the arg.
    extra_help: str, the extra_help to describe the resource. This should start
      with the verb, such as 'to update', that is acting on the resource.
    required: bool, if the arg is required.
    plural: bool, if True, use a resource argument that returns a list.

  Returns:
    the PresentationSpec for the resource argument.
  """
    if plural:
        help_stem = 'Names of one or more Notification Channels '
    else:
        help_stem = 'Name of the Notification Channel '
    return presentation_specs.ResourcePresentationSpec(arg_name, GetNotificationChannelResourceSpec(), help_stem + extra_help, required=required, plural=plural)