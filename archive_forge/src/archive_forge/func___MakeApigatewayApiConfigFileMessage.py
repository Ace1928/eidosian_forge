from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import os
import cloudsdk.google.protobuf.descriptor_pb2 as descriptor
from googlecloudsdk.api_lib.api_gateway import api_configs as api_configs_client
from googlecloudsdk.api_lib.api_gateway import apis as apis_client
from googlecloudsdk.api_lib.api_gateway import base as apigateway_base
from googlecloudsdk.api_lib.api_gateway import operations as operations_client
from googlecloudsdk.api_lib.endpoints import services_util as endpoints
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import base
from googlecloudsdk.calliope import exceptions as calliope_exceptions
from googlecloudsdk.command_lib.api_gateway import common_flags
from googlecloudsdk.command_lib.api_gateway import operations_util
from googlecloudsdk.command_lib.api_gateway import resource_args
from googlecloudsdk.command_lib.util.args import labels_util
from googlecloudsdk.core import log
from googlecloudsdk.core.util import http_encoding
def __MakeApigatewayApiConfigFileMessage(self, file_contents, filename, is_binary=False):
    """Constructs a ConfigFile message from a config file.

    Args:
      file_contents: The contents of the config file.
      filename: The path to the config file.
      is_binary: If set to true, the file_contents won't be encoded.

    Returns:
      The constructed ApigatewayApiConfigFile message.
    """
    messages = apigateway_base.GetMessagesModule()
    if not is_binary:
        file_contents = http_encoding.Encode(file_contents)
    return messages.ApigatewayApiConfigFile(contents=file_contents, path=os.path.basename(filename))