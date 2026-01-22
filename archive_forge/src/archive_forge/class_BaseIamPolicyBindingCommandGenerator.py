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
class BaseIamPolicyBindingCommandGenerator(BaseCommandGenerator):
    """Base class for iam binding command generators."""

    @property
    def _add_condition(self):
        return self.spec.iam and self.spec.iam.enable_condition

    @property
    def _hide_special_member_types(self):
        return self.spec.iam and self.spec.iam.hide_special_member_types

    def _GetResourceRef(self, args):
        methods = self._GetRuntimeMethods(args)
        return self.arg_generator.GetPrimaryResource(methods, args).Parse(args)

    def _GenerateDeclarativeIamRolesCompleter(self):
        """Generate a IAM role completer."""
        get_resource_ref = self._GetResourceRef

        class Completer(DeclarativeIamRolesCompleter):

            def __init__(self, **kwargs):
                super(Completer, self).__init__(get_resource_ref=get_resource_ref, **kwargs)
        return Completer

    def _GetIamPolicy(self, args):
        """GetIamPolicy helper function for add/remove binding."""
        get_iam_methods = self._GetMethods('getIamPolicy')
        get_iam_method = self.arg_generator.GetPrimaryResource(get_iam_methods, args).method
        get_iam_request = self.arg_generator.CreateRequest(args, get_iam_method)
        if self.spec.iam and self.spec.iam.policy_version:
            arg_utils.SetFieldInMessage(get_iam_request, self.spec.iam.get_iam_policy_version_path, self.spec.iam.policy_version)
        policy = get_iam_method.Call(get_iam_request)
        return policy