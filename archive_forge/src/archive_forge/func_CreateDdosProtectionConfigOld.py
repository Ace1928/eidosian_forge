from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def CreateDdosProtectionConfigOld(client, args, existing_ddos_protection_config):
    """Returns a SecurityPolicyDdosProtectionConfig message."""
    messages = client.messages
    ddos_protection_config = existing_ddos_protection_config if existing_ddos_protection_config is not None else messages.SecurityPolicyDdosProtectionConfig()
    if args.IsSpecified('ddos_protection'):
        ddos_protection_config.ddosProtection = messages.SecurityPolicyDdosProtectionConfig.DdosProtectionValueValuesEnum(args.ddos_protection)
    return ddos_protection_config