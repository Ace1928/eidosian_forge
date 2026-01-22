from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import json
import sys
from apitools.base.protorpclite import messages as apitools_messages
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py.exceptions import HttpBadRequestError
from googlecloudsdk.api_lib.util import waiter
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import command_loading
from googlecloudsdk.command_lib.util import completers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.command_lib.util.apis import yaml_command_schema
from googlecloudsdk.command_lib.util.apis import yaml_command_schema_util
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import resources
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_transform
from googlecloudsdk.core.util import files
import six
class DeclarativeIamRolesCompleter(completers.ListCommandCompleter):
    """An IAM role completer for a resource argument.

  The Complete() method override bypasses the completion cache.

  Attributes:
    _get_resource_ref: DeclarativeArgumentGenerator.GetPrimaryResource method
      to parse the resource ref.
  """

    def __init__(self, get_resource_ref, **kwargs):
        super(DeclarativeIamRolesCompleter, self).__init__(**kwargs)
        self._get_resource_ref = get_resource_ref

    def GetListCommand(self, parameter_info):
        resource_ref = self._get_resource_ref(parameter_info.parsed_args)
        resource_uri = resource_ref.SelfLink()
        return ['iam', 'list-grantable-roles', '--quiet', '--flatten=name', '--format=disable', resource_uri]

    def Complete(self, prefix, parameter_info):
        """Bypasses the cache and returns completions matching prefix."""
        command = self.GetListCommand(parameter_info)
        items = self.GetAllItems(command, parameter_info)
        return [item for item in items or [] if item is not None and item.startswith(prefix)]