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
class GetIamPolicyCommandGenerator(BaseCommandGenerator):
    """Generator for get-iam-policy commands."""
    command_type = yaml_command_schema.CommandType.GET_IAM_POLICY

    def _Generate(self):
        """Generates a get-iam-policy command.

    A get-iam-policy command has a single resource argument and an API method
    to call to get the resource. The result is returned using the default
    output format.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """
        from googlecloudsdk.command_lib.iam import iam_util

        class Command(base.ListCommand):
            """Get IAM policy command closure."""

            @staticmethod
            def Args(parser):
                self._CommonArgs(parser)
                base.URI_FLAG.RemoveFromParser(parser)

            def Run(self_, args):
                if self.spec.iam and self.spec.iam.policy_version:
                    self.spec.request.static_fields[self.spec.iam.get_iam_policy_version_path] = self.spec.iam.policy_version
                _, response = self._CommonRun(args)
                return self._HandleResponse(response, args)
        return Command