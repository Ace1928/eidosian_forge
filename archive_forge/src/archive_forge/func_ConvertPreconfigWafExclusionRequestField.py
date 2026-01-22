from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def ConvertPreconfigWafExclusionRequestField(request_field_in_rule, messages):
    """Converts the request field in preconfigured WAF exclusion configuration.

  Args:
    request_field_in_rule: a request field in preconfigured WAF exclusion
      configuration read from the security policy config file.
    messages: the set of available messages.

  Returns:
    The proto representation of the request field.
  """
    request_field = messages.SecurityPolicyRulePreconfiguredWafConfigExclusionFieldParams()
    if 'op' in request_field_in_rule:
        request_field.op = messages.SecurityPolicyRulePreconfiguredWafConfigExclusionFieldParams.OpValueValuesEnum(request_field_in_rule['op'])
    if 'val' in request_field_in_rule:
        request_field.val = request_field_in_rule['val']
    return request_field