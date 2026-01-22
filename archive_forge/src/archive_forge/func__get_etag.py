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
def _get_etag(object_dict):
    """Returns the cleaned-up etag value, if present."""
    if 'ETag' in object_dict:
        etag = object_dict.get('ETag')
    elif 'CopyObjectResult' in object_dict:
        etag = object_dict['CopyObjectResult'].get('ETag')
    else:
        etag = None
    if etag and etag.startswith('"') and etag.endswith('"'):
        return etag.strip('"')
    return etag