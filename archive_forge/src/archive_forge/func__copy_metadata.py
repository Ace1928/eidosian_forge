from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import copy
import re
from googlecloudsdk.api_lib.storage import metadata_util
from googlecloudsdk.api_lib.storage import xml_metadata_field_converters
from googlecloudsdk.command_lib.storage import errors
from googlecloudsdk.command_lib.storage import storage_url
from googlecloudsdk.command_lib.storage import user_request_args_factory
from googlecloudsdk.command_lib.storage.resources import gcs_resource_reference
from googlecloudsdk.command_lib.storage.resources import resource_reference
from googlecloudsdk.command_lib.storage.resources import s3_resource_reference
from googlecloudsdk.core import log
def _copy_metadata(source_metadata_dict, destination_metadata_dict, fields):
    """Copy fields(provided in arguments) from one metadata dict to another."""
    if not destination_metadata_dict:
        destination_metadata_dict = {}
    if not source_metadata_dict:
        return destination_metadata_dict
    for field in fields:
        if field in source_metadata_dict:
            destination_metadata_dict[field] = source_metadata_dict[field]
    return destination_metadata_dict