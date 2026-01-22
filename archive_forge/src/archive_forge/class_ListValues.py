from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.resourcesettings import utils as api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_settings import arguments
from googlecloudsdk.command_lib.resource_settings import utils
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class ListValues(base.ListCommand):
    """List the values for any configured resource settings.

  List the values for any configured resource settings.

  ## EXAMPLES

  To list all setting values on the project ``foo-project'', run:

    $ {command} --project=foo-project
  """

    @staticmethod
    def Args(parser):
        arguments.AddResourceFlagsToParser(parser)

    def Run(self, args):
        """List all setting values.

    Args:
      args: argparse.Namespace, An object that contains the values for the
        arguments specified in the Args method.

    Returns:
       The list of setting values.
    """
        settings_service = api_utils.GetServiceFromArgs(args)
        parent_resource = utils.GetParentResourceFromArgs(args)
        get_request = api_utils.GetListRequestFromArgs(args, parent_resource, True)
        setting_value = settings_service.List(get_request)
        return setting_value