from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import collections
import os
from apitools.base.protorpclite import messages
from googlecloudsdk.command_lib.anthos.common import file_parsers
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.command_lib.util.apis import registry
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import yaml
from googlecloudsdk.core import yaml_validator
from googlecloudsdk.core.util import files
import six
def GenerateMessageMappingFromFile(input_file):
    """Build apitools to krm mapping from a YAML/JSON File."""
    config_file = file_parsers.YamlConfigFile(ApitoolsToKrmConfigObject, file_path=input_file)
    config_data = config_file.data[0]
    ValidateMessageMappingFile(config_data.content)
    request_type = config_data.apitools_request
    mapping = collections.OrderedDict()
    for msg_field, value in six.iteritems(config_data):
        mapping[msg_field] = ApitoolsToKrmFieldDescriptor.FromYamlData(msg_field, value)
    return (request_type, mapping)