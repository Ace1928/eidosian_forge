from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import os
from apitools.base.protorpclite import messages
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
def GenerateMappingFileTemplate(api_name, message_type, skip_fields=None, file_path=None, api_version=None, known_mappings=None):
    """Create a stub Apitools To KRM mapping file for specified Apitools message.

  Args:
      api_name: string, The api containing the message.
      message_type: string, The message to generate mapping for.
      skip_fields: [string], A list of field paths to exclude from mapping file.
      file_path: string, path of destination file. If None, will write result to
        stdout.
      api_version: Version of the api to retrieve the message type from. If None
        will use default API version.
      known_mappings: {string: object}, Fields to pre-initialize in the mapping.

  Returns:
    The path to the created file or file contents if no path specified.
  Raises:
    InvalidDataError, if api or message are invalid.
  """
    try:
        api_obj = registry.GetAPI(api_name, api_version)
        all_messages = api_obj.GetMessagesModule()
        message = getattr(all_messages, message_type)
        mapping_object = _BuildYamlMappingTemplateFromMessage(message)
        if skip_fields:
            for path in skip_fields:
                file_parsers.DeleteItemInDict(mapping_object, path)
        if known_mappings:
            for path, value in six.iteritems(known_mappings):
                file_parsers.FindOrSetItemInDict(mapping_object, path, set_value=value)
        yaml.convert_to_block_text(mapping_object)
        output = yaml.dump(mapping_object, round_trip=True)
        if file_path:
            files.WriteFileAtomically(file_path, output)
            output = file_path
        return output
    except (AttributeError, registry.Error) as ae:
        raise InvalidDataError('Error retrieving message [{message}] from API [{api}/{ver}] :: {error}'.format(message=message_type, api=api_name, ver=api_version or 'default', error=ae))