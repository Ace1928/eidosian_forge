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
class CorsTranslation(object):
    """Functions for converting between various CORS formats.

    This class handles conversation to and from Boto Cors objects, JSON text,
    and apitools Message objects.
  """

    @classmethod
    def BotoCorsFromMessage(cls, cors_message):
        """Translates an apitools message to a boto Cors object."""
        cors = boto.gs.cors.Cors()
        cors.cors = []
        for collection_message in cors_message:
            collection_elements = []
            if collection_message.maxAgeSeconds:
                collection_elements.append((boto.gs.cors.MAXAGESEC, str(collection_message.maxAgeSeconds)))
            if collection_message.method:
                method_elements = []
                for method in collection_message.method:
                    method_elements.append((boto.gs.cors.METHOD, method))
                collection_elements.append((boto.gs.cors.METHODS, method_elements))
            if collection_message.origin:
                origin_elements = []
                for origin in collection_message.origin:
                    origin_elements.append((boto.gs.cors.ORIGIN, origin))
                collection_elements.append((boto.gs.cors.ORIGINS, origin_elements))
            if collection_message.responseHeader:
                header_elements = []
                for header in collection_message.responseHeader:
                    header_elements.append((boto.gs.cors.HEADER, header))
                collection_elements.append((boto.gs.cors.HEADERS, header_elements))
            cors.cors.append(collection_elements)
        return cors

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

    @classmethod
    def JsonCorsToMessageEntries(cls, json_cors):
        """Translates CORS JSON to an apitools message.

    Args:
      json_cors: JSON string representing CORS configuration.

    Raises:
      ArgumentException on invalid CORS JSON data.

    Returns:
      List of apitools Bucket.CorsValueListEntry. An empty list represents
      no CORS configuration.
    """
        deserialized_cors = None
        try:
            deserialized_cors = json.loads(json_cors)
        except ValueError:
            CheckForXmlConfigurationAndRaise('CORS', json_cors)
        if not isinstance(deserialized_cors, list):
            raise ArgumentException('CORS JSON should be formatted as a list containing one or more JSON objects.\nSee "gsutil help cors".')
        cors = []
        for cors_entry in deserialized_cors:
            cors.append(encoding.DictToMessage(cors_entry, apitools_messages.Bucket.CorsValueListEntry))
        return cors

    @classmethod
    def MessageEntriesToJson(cls, cors_message):
        """Translates an apitools message to CORS JSON."""
        json_text = ''
        json_text += '['
        printed_one = False
        for cors_entry in cors_message:
            if printed_one:
                json_text += ','
            else:
                printed_one = True
            json_text += encoding.MessageToJson(cors_entry)
        json_text += ']\n'
        return json_text