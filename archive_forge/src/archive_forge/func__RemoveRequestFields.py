from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.security_policies import flags as security_policy_flags
from googlecloudsdk.command_lib.compute.security_policies.rules import flags
from googlecloudsdk.core import properties
@classmethod
def _RemoveRequestFields(cls, existing_request_fields, request_fields_to_remove):
    new_request_fields = []
    for existing_request_field in existing_request_fields:
        if existing_request_field not in request_fields_to_remove:
            new_request_fields.append(existing_request_field)
    return new_request_fields