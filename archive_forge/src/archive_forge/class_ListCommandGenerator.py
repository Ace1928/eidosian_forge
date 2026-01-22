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
class ListCommandGenerator(BaseCommandGenerator):
    """Generator for list commands."""
    command_type = yaml_command_schema.CommandType.LIST

    def _Generate(self):
        """Generates a List command.

    A list command operates on a single resource and has flags for the parent
    collection of that resource. Because it extends the calliope base List
    command, it gets flags for things like limit, filter, and page size. A
    list command should register a table output format to display the result.
    If arguments.resource.response_id_field is specified, a --uri flag will also
    be enabled.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """

        class Command(base.ListCommand):

            @staticmethod
            def Args(parser):
                self._CommonArgs(parser)
                if not self.spec.response.id_field:
                    base.URI_FLAG.RemoveFromParser(parser)

            def Run(self_, args):
                self._RegisterURIFunc(args)
                unused_ref, response = self._CommonRun(args)
                return self._HandleResponse(response, args)
        return Command