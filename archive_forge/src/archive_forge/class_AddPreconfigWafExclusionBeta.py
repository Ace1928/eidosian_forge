from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from apitools.base.py import encoding
from googlecloudsdk.api_lib.compute import base_classes
from googlecloudsdk.api_lib.compute.security_policies import client
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.command_lib.compute import scope as compute_scope
from googlecloudsdk.command_lib.compute.security_policies import flags as security_policy_flags
from googlecloudsdk.command_lib.compute.security_policies.rules import flags
from googlecloudsdk.core import properties
from googlecloudsdk.core import resources
@base.ReleaseTracks(base.ReleaseTrack.BETA)
class AddPreconfigWafExclusionBeta(AddPreconfigWafExclusionGA):
    """Add an exclusion configuration for preconfigured WAF evaluation into a security policy rule.

  *{command}* is used to add an exclusion configuration for preconfigured WAF
  evaluation into a security policy rule.

  Note that request field exclusions are associated with a target, which can be
  a single rule set, or a rule set plus a list of rule IDs under the rule set.

  ## EXAMPLES

  To add specific request field exclusions that are associated with the target
  of 'sqli-stable': ['owasp-crs-v030001-id942110-sqli',
  'owasp-crs-v030001-id942120-sqli'], run:

    $ {command} 1000 \\
       --security-policy=my-policy \\
       --target-rule-set=sqli-stable \\
       --target-rule-ids=owasp-crs-v030001-id942110-sqli,owasp-crs-v030001-id942120-sqli
       \\
       --request-header-to-exclude=op=EQUALS,val=abc \\
       --request-header-to-exclude=op=STARTS_WITH,val=xyz \\
       --request-uri-to-exclude=op=EQUALS_ANY

  To add specific request field exclusions that are associated with the target
  of 'sqli-stable': [], run:

    $ {command} 1000 \\
       --security-policy=my-policy \\
       --target-rule-set=sqli-stable \\
       --request-cookie-to-exclude=op=EQUALS_ANY
  """