from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import json
from googlecloudsdk.calliope import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core.resource import resource_printer
import six
def CreateAdvancedOptionsConfig(client, args, existing_advanced_options_config):
    """Returns a SecurityPolicyAdvancedOptionsConfig message."""
    messages = client.messages
    advanced_options_config = existing_advanced_options_config if existing_advanced_options_config is not None else messages.SecurityPolicyAdvancedOptionsConfig()
    if args.IsSpecified('json_parsing'):
        advanced_options_config.jsonParsing = messages.SecurityPolicyAdvancedOptionsConfig.JsonParsingValueValuesEnum(args.json_parsing)
    if args.IsSpecified('json_custom_content_types'):
        advanced_options_config.jsonCustomConfig = messages.SecurityPolicyAdvancedOptionsConfigJsonCustomConfig(contentTypes=args.json_custom_content_types)
    if args.IsSpecified('log_level'):
        advanced_options_config.logLevel = messages.SecurityPolicyAdvancedOptionsConfig.LogLevelValueValuesEnum(args.log_level)
    if args.IsSpecified('user_ip_request_headers'):
        advanced_options_config.userIpRequestHeaders = args.user_ip_request_headers
    return advanced_options_config