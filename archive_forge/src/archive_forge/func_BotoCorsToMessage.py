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
@classmethod
def BotoCorsToMessage(cls, boto_cors):
    """Translates a boto Cors object to an apitools message."""
    message_cors = []
    if boto_cors.cors:
        for cors_collection in boto_cors.cors:
            if cors_collection:
                collection_message = apitools_messages.Bucket.CorsValueListEntry()
                for element_tuple in cors_collection:
                    if element_tuple[0] == boto.gs.cors.MAXAGESEC:
                        collection_message.maxAgeSeconds = int(element_tuple[1])
                    if element_tuple[0] == boto.gs.cors.METHODS:
                        for method_tuple in element_tuple[1]:
                            collection_message.method.append(method_tuple[1])
                    if element_tuple[0] == boto.gs.cors.ORIGINS:
                        for origin_tuple in element_tuple[1]:
                            collection_message.origin.append(origin_tuple[1])
                    if element_tuple[0] == boto.gs.cors.HEADERS:
                        for header_tuple in element_tuple[1]:
                            collection_message.responseHeader.append(header_tuple[1])
                message_cors.append(collection_message)
    return message_cors