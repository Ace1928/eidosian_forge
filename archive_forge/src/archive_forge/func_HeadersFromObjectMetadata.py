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
def HeadersFromObjectMetadata(dst_obj_metadata, provider):
    """Creates a header dictionary based on existing object metadata.

  Args:
    dst_obj_metadata: Object metadata to create the headers from.
    provider: Provider string ('gs' or 's3').

  Returns:
    Headers dictionary.
  """
    headers = {}
    if not dst_obj_metadata:
        return
    if dst_obj_metadata.cacheControl is not None:
        if not dst_obj_metadata.cacheControl:
            headers['cache-control'] = None
        else:
            headers['cache-control'] = dst_obj_metadata.cacheControl.strip()
    if dst_obj_metadata.contentDisposition:
        if not dst_obj_metadata.contentDisposition:
            headers['content-disposition'] = None
        else:
            headers['content-disposition'] = dst_obj_metadata.contentDisposition.strip()
    if dst_obj_metadata.contentEncoding:
        if not dst_obj_metadata.contentEncoding:
            headers['content-encoding'] = None
        else:
            headers['content-encoding'] = dst_obj_metadata.contentEncoding.strip()
    if dst_obj_metadata.contentLanguage:
        if not dst_obj_metadata.contentLanguage:
            headers['content-language'] = None
        else:
            headers['content-language'] = dst_obj_metadata.contentLanguage.strip()
    if dst_obj_metadata.md5Hash:
        if not dst_obj_metadata.md5Hash:
            headers['Content-MD5'] = None
        else:
            headers['Content-MD5'] = dst_obj_metadata.md5Hash.strip()
    if dst_obj_metadata.contentType is not None:
        if not dst_obj_metadata.contentType:
            headers['content-type'] = None
        else:
            headers['content-type'] = dst_obj_metadata.contentType.strip()
    if dst_obj_metadata.customTime is not None:
        if not dst_obj_metadata.customTime:
            headers['custom-time'] = None
        else:
            headers['custom-time'] = dst_obj_metadata.customTime.strip()
    if dst_obj_metadata.storageClass:
        header_name = 'storage-class'
        if provider == 'gs':
            header_name = 'x-goog-' + header_name
        elif provider == 's3':
            header_name = 'x-amz-' + header_name
        else:
            raise ArgumentException('Invalid provider specified: %s' % provider)
        headers[header_name] = dst_obj_metadata.storageClass.strip()
    if dst_obj_metadata.metadata and dst_obj_metadata.metadata.additionalProperties:
        for additional_property in dst_obj_metadata.metadata.additionalProperties:
            if additional_property.key == 'content-language':
                continue
            if additional_property.key in S3_MARKER_GUIDS:
                continue
            if provider == 'gs':
                header_name = 'x-goog-meta-' + additional_property.key
            elif provider == 's3':
                if additional_property.key.startswith(S3_HEADER_PREFIX):
                    header_name = 'x-amz-' + additional_property.key[len(S3_HEADER_PREFIX):]
                else:
                    header_name = 'x-amz-meta-' + additional_property.key
            else:
                raise ArgumentException('Invalid provider specified: %s' % provider)
            if additional_property.value is not None and (not additional_property.value):
                headers[header_name] = None
            else:
                headers[header_name] = additional_property.value
    return headers