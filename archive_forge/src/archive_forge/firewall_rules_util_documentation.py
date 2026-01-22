from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import resources
import six
Converts an action string to the corresponding enum value.

  Options are: 'allow' or 'deny', otherwise None will be returned.

  Args:
    messages: apitools.base.protorpclite.messages, the proto messages class for
      this API version for firewall.
    action: str, the action as a string
  Returns:
    ActionValueValuesEnum type
  