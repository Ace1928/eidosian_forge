from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import io
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute import property_selector
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.compute import exceptions
from googlecloudsdk.command_lib.compute import flags as compute_flags
from googlecloudsdk.command_lib.compute.backend_services import backend_services_utils
from googlecloudsdk.command_lib.compute.backend_services import flags
from googlecloudsdk.core import resources
from googlecloudsdk.core import yaml
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import edit
import six
def EditResource(self, args, backend_service_ref, file_contents, holder, modifiable_record, original_object, original_record):
    while True:
        try:
            file_contents = edit.OnlineEdit(file_contents)
        except edit.NoSaveException:
            raise exceptions.AbortedError('Edit aborted by user.')
        try:
            resource_list = self._ProcessEditedResource(holder, backend_service_ref, file_contents, original_object, original_record, modifiable_record, args)
            break
        except (ValueError, yaml.YAMLParseError, messages.ValidationError, calliope_exceptions.ToolException) as e:
            message = getattr(e, 'message', six.text_type(e))
            if isinstance(e, calliope_exceptions.ToolException):
                problem_type = 'applying'
            else:
                problem_type = 'parsing'
            message = 'There was a problem {0} your changes: {1}'.format(problem_type, message)
            if not console_io.PromptContinue(message=message, prompt_string='Would you like to edit the resource again?'):
                raise exceptions.AbortedError('Edit aborted by user.')
    return resource_list