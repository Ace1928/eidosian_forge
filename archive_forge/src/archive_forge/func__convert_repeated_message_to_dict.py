from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import copy
import datetime
import sys
from cloudsdk.google.protobuf import json_format
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.storage.gcs_json import metadata_util as json_metadata_util
from googlecloudsdk.command_lib.storage import encryption_util
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import hash_util
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.util import crc32c
def _convert_repeated_message_to_dict(message):
    """Converts a sequence of proto messages to a dict."""
    if not message:
        return
    return [json_format.MessageToDict(i._pb) for i in message]