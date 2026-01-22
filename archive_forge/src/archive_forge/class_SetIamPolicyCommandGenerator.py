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
class SetIamPolicyCommandGenerator(BaseCommandGenerator):
    """Generator for set-iam-policy commands."""
    command_type = yaml_command_schema.CommandType.SET_IAM_POLICY

    def _SetPolicyUpdateMask(self, update_mask, method):
        """Set Field Mask on SetIamPolicy request message.

    If the API supports update_masks then adds the update_mask to the
    SetIamPolicy request (via static fields).

    Args:
      update_mask: str, comma separated string listing the Policy fields to be
        updated.
      method: APIMethod, used to identify update mask field.
    """
        set_iam_policy_request = 'SetIamPolicyRequest'
        policy_request_path = 'setIamPolicyRequest'
        if self.spec.iam:
            overrides = self.spec.iam.message_type_overrides
            if 'set_iam_policy_request' in overrides:
                set_iam_policy_request = overrides['set_iam_policy_request'] or set_iam_policy_request
            policy_request_path = self.spec.iam.set_iam_policy_request_path or policy_request_path
        mask_field_path = '{}.updateMask'.format(policy_request_path)
        update_request = method.GetMessageByName(set_iam_policy_request)
        if hasattr(update_request, 'updateMask'):
            self.spec.request.static_fields[mask_field_path] = update_mask

    def _Generate(self):
        """Generates a set-iam-policy command.

    A set-iam-policy command takes a resource argument, a policy to set on that
    resource, and an API method to call to set the policy on the resource. The
    result is returned using the default output format.

    Returns:
      calliope.base.Command, The command that implements the spec.
    """
        from googlecloudsdk.command_lib.iam import iam_util

        class Command(base.Command):
            """Set IAM policy command closure."""

            @staticmethod
            def Args(parser):
                self._CommonArgs(parser)
                iam_util.AddArgForPolicyFile(parser)
                base.URI_FLAG.RemoveFromParser(parser)

            def Run(self_, args):
                """Called when command is executed."""
                policy_type_name = 'Policy'
                policy_request_path = 'setIamPolicyRequest'
                if self.spec.iam:
                    if 'policy' in self.spec.iam.message_type_overrides:
                        policy_type_name = self.spec.iam.message_type_overrides['policy'] or policy_type_name
                    policy_request_path = self.spec.iam.set_iam_policy_request_path or policy_request_path
                policy_field_path = policy_request_path + '.policy'
                method = self.arg_generator.GetPrimaryResource(self.methods, args).method
                policy_type = method.GetMessageByName(policy_type_name)
                if not policy_type:
                    raise ValueError('Policy type [{}] not found.'.format(policy_type_name))
                policy, update_mask = iam_util.ParsePolicyFileWithUpdateMask(args.policy_file, policy_type)
                if self.spec.iam and self.spec.iam.policy_version:
                    policy.version = self.spec.iam.policy_version
                self.spec.request.static_fields[policy_field_path] = policy
                self._SetPolicyUpdateMask(update_mask, method)
                try:
                    ref, response = self._CommonRun(args)
                except HttpBadRequestError as ex:
                    log.err.Print('ERROR: Policy modification failed. For bindings with conditions, run "gcloud alpha iam policies lint-condition" to identify issues in conditions.')
                    raise ex
                iam_util.LogSetIamPolicy(ref.Name(), self._GetDisplayResourceType(args))
                return self._HandleResponse(response, args)
        return Command