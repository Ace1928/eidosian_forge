from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def CreateUserDefinedField(client, args):
    """Returns a SecurityPolicyUserDefinedField message."""
    messages = client.messages
    user_defined_field = messages.SecurityPolicyUserDefinedField()
    user_defined_field.name = args.user_defined_field_name
    user_defined_field.base = messages.SecurityPolicyUserDefinedField.BaseValueValuesEnum(_ConvertUserDefinedFieldBase(args.base))
    user_defined_field.offset = args.offset
    user_defined_field.size = args.size
    if getattr(args, 'mask', None) is not None:
        user_defined_field.mask = args.mask
    return user_defined_field