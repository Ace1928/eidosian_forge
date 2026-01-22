from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import exceptions as api_exceptions
from argcomplete import completers
from googlecloudsdk.api_lib.resourcesettings import service
from googlecloudsdk.api_lib.resourcesettings import utils as api_utils
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.resource_settings import exceptions
from googlecloudsdk.command_lib.resource_settings import utils
@base.ReleaseTracks(base.ReleaseTrack.ALPHA, base.ReleaseTrack.GA)
class SetValue(base.Command):
    """Update the value of a resource setting.

  Update the value of a resource setting.

  ## EXAMPLES

  To set the resource setting from the file with the path ``./sample_path'',
  run:

    $ {command} --value-file="./test_input.json"
  """

    @staticmethod
    def Args(parser):
        parser.add_argument('--value-file', metavar='value-file', completer=completers.FilesCompleter, required=True, help='Path to JSON or YAML file that contains the resource setting.')

    def Run(self, args):
        """Creates or updates a setting from a JSON or YAML file.

    Args:
      args: argparse.Namespace, An object that contains the values for the
        arguments specified in the Args method.

    Returns:
      The created or updated setting.
    """
        settings_message = service.ResourceSettingsMessages()
        input_setting = utils.GetMessageFromFile(args.value_file, settings_message.GoogleCloudResourcesettingsV1Setting)
        if not input_setting.name:
            raise exceptions.InvalidInputError('Name field not present in the resource setting.')
        if not utils.ValidateSettingPath(input_setting.name):
            raise exceptions.InvalidInputError('Name field has invalid syntax')
        resource_type = utils.GetResourceTypeFromString(input_setting.name)
        settings_service = api_utils.GetServiceFromResourceType(resource_type)
        setting_path = input_setting.name
        etag = input.etag if hasattr(input, 'etag') else None
        update_request = api_utils.GetPatchRequestFromResourceType(resource_type, setting_path, input_setting.localValue, etag)
        update_response = settings_service.Patch(update_request)
        return update_response