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
def _ConfigureCommand(self, command):
    """Configures top level attributes of the generated command.

    Args:
      command: The command being generated.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """
    if self.spec.hidden:
        command = base.Hidden(command)
    if self.spec.universe_compatible is not None:
        if self.spec.universe_compatible:
            command = base.UniverseCompatible(command)
        else:
            command = base.DefaultUniverseOnly(command)
    if self.spec.release_tracks:
        command = base.ReleaseTracks(*self.spec.release_tracks)(command)
    if self.spec.deprecated_data:
        command = base.Deprecate(**self.spec.deprecated_data)(command)
    if not hasattr(command, 'detailed_help'):
        key_map = {'description': 'DESCRIPTION', 'examples': 'EXAMPLES'}
        command.detailed_help = {key_map.get(k, k): v for k, v in self.spec.help_text.items()}
    if self.has_request_method:
        api_names = set((f'{method.collection.api_name}/{method.collection.api_version}' for method in self.methods))
        doc_urls = set((method.collection.docs_url for method in self.methods))
        api_name_str = ', '.join(api_names)
        doc_url_str = ', '.join(doc_urls)
        if len(api_names) > 1:
            api_info = f'This command uses *{api_name_str}* APIs. The full documentation for these APIs can be found at: {doc_url_str}'
        else:
            api_info = f'This command uses the *{api_name_str}* API. The full documentation for this API can be found at: {doc_url_str}'
        command.detailed_help['API REFERENCE'] = api_info
    return command