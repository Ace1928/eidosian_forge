from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.org_security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.compute.org_security_policies import flags
from googlecloudsdk.command_lib.compute.org_security_policies import org_security_policies_utils
import six
Replace the rules of a Compute Engine organization security policy with rules from another policy.

  *{command}* is used to replace the rules of organization security policies. An
   organization security policy is a set of rules that controls access to
   various resources.
  