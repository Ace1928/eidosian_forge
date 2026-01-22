from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import datetime
import json
import re
import textwrap
import xml.etree.ElementTree
from xml.etree.ElementTree import ParseError as XmlParseError
import six
from apitools.base.protorpclite.util import decode_datetime
from apitools.base.py import encoding
import boto
from boto.gs.acl import ACL
from boto.gs.acl import ALL_AUTHENTICATED_USERS
from boto.gs.acl import ALL_USERS
from boto.gs.acl import Entries
from boto.gs.acl import Entry
from boto.gs.acl import GROUP_BY_DOMAIN
from boto.gs.acl import GROUP_BY_EMAIL
from boto.gs.acl import GROUP_BY_ID
from boto.gs.acl import USER_BY_EMAIL
from boto.gs.acl import USER_BY_ID
from boto.s3.tagging import Tags
from boto.s3.tagging import TagSet
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import BucketNotFoundException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import Preconditions
from gslib.exception import CommandException
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils.constants import S3_ACL_MARKER_GUID
from gslib.utils.constants import S3_MARKER_GUIDS
def ObjectMetadataFromHeaders(headers):
    """Creates object metadata according to the provided headers.

  gsutil -h allows specifiying various headers (originally intended
  to be passed to boto in gsutil v3).  For the JSON API to be compatible with
  this option, we need to parse these headers into gsutil_api Object fields.

  Args:
    headers: Dict of headers passed via gsutil -h

  Raises:
    ArgumentException if an invalid header is encountered.

  Returns:
    apitools Object with relevant fields populated from headers.
  """
    obj_metadata = apitools_messages.Object()
    for header, value in headers.items():
        if CACHE_CONTROL_REGEX.match(header):
            obj_metadata.cacheControl = value.strip()
        elif CONTENT_DISPOSITION_REGEX.match(header):
            obj_metadata.contentDisposition = value.strip()
        elif CONTENT_ENCODING_REGEX.match(header):
            obj_metadata.contentEncoding = value.strip()
        elif CONTENT_MD5_REGEX.match(header):
            obj_metadata.md5Hash = value.strip()
        elif CONTENT_LANGUAGE_REGEX.match(header):
            obj_metadata.contentLanguage = value.strip()
        elif CONTENT_TYPE_REGEX.match(header):
            if not value:
                obj_metadata.contentType = DEFAULT_CONTENT_TYPE
            else:
                obj_metadata.contentType = value.strip()
        elif CUSTOM_TIME_REGEX.match(header):
            obj_metadata.customTime = decode_datetime(value.strip())
        elif GOOG_API_VERSION_REGEX.match(header):
            continue
        elif GOOG_GENERATION_MATCH_REGEX.match(header):
            continue
        elif GOOG_METAGENERATION_MATCH_REGEX.match(header):
            continue
        else:
            custom_goog_metadata_match = CUSTOM_GOOG_METADATA_REGEX.match(header)
            custom_amz_metadata_match = CUSTOM_AMZ_METADATA_REGEX.match(header)
            custom_amz_header_match = CUSTOM_AMZ_HEADER_REGEX.match(header)
            header_key = None
            if custom_goog_metadata_match:
                header_key = custom_goog_metadata_match.group('header_key')
            elif custom_amz_metadata_match:
                header_key = custom_amz_metadata_match.group('header_key')
            elif custom_amz_header_match:
                header_key = S3_HEADER_PREFIX + custom_amz_header_match.group('header_key')
            if header_key:
                if header_key.lower() == 'x-goog-content-language':
                    continue
                if not obj_metadata.metadata:
                    obj_metadata.metadata = apitools_messages.Object.MetadataValue()
                if not obj_metadata.metadata.additionalProperties:
                    obj_metadata.metadata.additionalProperties = []
                obj_metadata.metadata.additionalProperties.append(apitools_messages.Object.MetadataValue.AdditionalProperty(key=header_key, value=value))
    return obj_metadata