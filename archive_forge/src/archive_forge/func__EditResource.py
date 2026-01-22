from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import property_selector
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import exceptions as compute_exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.url_maps import flags
from googlecloudsdk.command_lib.compute.url_maps import url_maps_utils
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import edit
import six
def _EditResource(args, client, holder, original_object, url_map_ref, track):
    """Allows user to edit the URL Map."""
    original_record = encoding.MessageToDict(original_object)
    field_selector = property_selector.PropertySelector(properties=['defaultService', 'description', 'hostRules', 'pathMatchers', 'tests'])
    modifiable_record = field_selector.Apply(original_record)
    buf = _BuildFileContents(args, client, modifiable_record, original_record, track)
    file_contents = buf.getvalue()
    while True:
        try:
            file_contents = edit.OnlineEdit(file_contents)
        except edit.NoSaveException:
            raise compute_exceptions.AbortedError('Edit aborted by user.')
        try:
            resource_list = _ProcessEditedResource(holder, url_map_ref, file_contents, original_object, original_record, modifiable_record, args)
            break
        except (ValueError, yaml.YAMLParseError, messages.ValidationError, exceptions.ToolException) as e:
            message = getattr(e, 'message', six.text_type(e))
            if isinstance(e, exceptions.ToolException):
                problem_type = 'applying'
            else:
                problem_type = 'parsing'
            message = 'There was a problem {0} your changes: {1}'.format(problem_type, message)
            if not console_io.PromptContinue(message=message, prompt_string='Would you like to edit the resource again?'):
                raise compute_exceptions.AbortedError('Edit aborted by user.')
    return resource_list