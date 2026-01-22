from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
import base64
import binascii
import datetime
import errno
import json
import os
import pickle
import random
import re
import socket
import tempfile
import textwrap
import threading
import time
import xml
from xml.dom.minidom import parseString as XmlParseString
from xml.sax import _exceptions as SaxExceptions
import six
from six.moves import http_client
import boto
from boto import config
from boto import handler
from boto.gs.cors import Cors
from boto.gs.lifecycle import LifecycleConfig
from boto.s3.cors import CORSConfiguration as S3Cors
from boto.s3.deletemarker import DeleteMarker
from boto.s3.lifecycle import Lifecycle as S3Lifecycle
from boto.s3.prefix import Prefix
from boto.s3.tagging import Tags
import boto.exception
import boto.utils
from gslib.boto_resumable_upload import BotoResumableUpload
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import CloudApi
from gslib.cloud_api import NotEmptyException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import ResumableDownloadException
from gslib.cloud_api import ResumableUploadAbortException
from gslib.cloud_api import ResumableUploadException
from gslib.cloud_api import ResumableUploadStartOverException
from gslib.cloud_api import ServiceException
import gslib.devshell_auth_plugin  # pylint: disable=unused-import
from gslib.exception import CommandException
from gslib.exception import InvalidUrlError
from gslib.project_id import GOOG_PROJ_ID_HDR
from gslib.project_id import PopulateProjectId
from gslib.storage_url import GenerationFromUrlAndString
from gslib.storage_url import StorageUrlFromString
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.utils import parallelism_framework_util
from gslib.utils.boto_util import ConfigureNoOpAuthIfNeeded
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNumRetries
from gslib.utils.cloud_api_helper import ListToGetFields
from gslib.utils.cloud_api_helper import ValidateDstObjectMetadata
from gslib.utils.constants import DEFAULT_FILE_BUFFER_SIZE
from gslib.utils.constants import REQUEST_REASON_ENV_VAR
from gslib.utils.constants import REQUEST_REASON_HEADER_KEY
from gslib.utils.constants import S3_DELETE_MARKER_GUID
from gslib.utils.constants import UTF8
from gslib.utils.constants import XML_PROGRESS_CALLBACKS
from gslib.utils.hashing_helper import Base64EncodeHash
from gslib.utils.hashing_helper import Base64ToHexHash
from gslib.utils.metadata_util import AddAcceptEncodingGzipIfNeeded
from gslib.utils.parallelism_framework_util import multiprocessing_context
from gslib.utils.text_util import EncodeStringAsLong
from gslib.utils.translation_helper import AclTranslation
from gslib.utils.translation_helper import AddS3MarkerAclToObjectMetadata
from gslib.utils.translation_helper import CorsTranslation
from gslib.utils.translation_helper import CreateBucketNotFoundException
from gslib.utils.translation_helper import CreateNotFoundExceptionForObjectWrite
from gslib.utils.translation_helper import CreateObjectNotFoundException
from gslib.utils.translation_helper import DEFAULT_CONTENT_TYPE
from gslib.utils.translation_helper import HeadersFromObjectMetadata
from gslib.utils.translation_helper import LabelTranslation
from gslib.utils.translation_helper import LifecycleTranslation
from gslib.utils.translation_helper import REMOVE_CORS_CONFIG
from gslib.utils.translation_helper import S3MarkerAclFromObjectMetadata
from gslib.utils.translation_helper import UnaryDictToXml
from gslib.utils.unit_util import TWO_MIB
def _BotoBucketToBucket(self, bucket, fields=None):
    """Constructs an apitools Bucket from a boto bucket.

    Args:
      bucket: Boto bucket.
      fields: If present, construct the apitools Bucket with only this set of
              metadata fields.

    Returns:
      apitools Bucket.
    """
    bucket_uri = self._StorageUriForBucket(bucket.name)
    cloud_api_bucket = apitools_messages.Bucket(name=bucket.name, id=bucket.name)
    headers = self._CreateBaseHeaders()
    if self.provider == 'gs':
        if not fields or 'storageClass' in fields:
            if hasattr(bucket, 'get_storage_class'):
                cloud_api_bucket.storageClass = bucket.get_storage_class()
        if not fields or 'acl' in fields:
            try:
                for acl in AclTranslation.BotoBucketAclToMessage(bucket.get_acl(headers=headers)):
                    cloud_api_bucket.acl.append(acl)
            except TRANSLATABLE_BOTO_EXCEPTIONS as e:
                translated_exception = self._TranslateBotoException(e, bucket_name=bucket.name)
                if translated_exception and isinstance(translated_exception, AccessDeniedException):
                    pass
                else:
                    self._TranslateExceptionAndRaise(e, bucket_name=bucket.name)
        if not fields or 'cors' in fields:
            try:
                boto_cors = bucket_uri.get_cors(headers=headers)
                cloud_api_bucket.cors = CorsTranslation.BotoCorsToMessage(boto_cors)
            except TRANSLATABLE_BOTO_EXCEPTIONS as e:
                self._TranslateExceptionAndRaise(e, bucket_name=bucket.name)
        if not fields or 'defaultObjectAcl' in fields:
            try:
                for acl in AclTranslation.BotoObjectAclToMessage(bucket.get_def_acl(headers=headers)):
                    cloud_api_bucket.defaultObjectAcl.append(acl)
            except TRANSLATABLE_BOTO_EXCEPTIONS as e:
                translated_exception = self._TranslateBotoException(e, bucket_name=bucket.name)
                if translated_exception and isinstance(translated_exception, AccessDeniedException):
                    pass
                else:
                    self._TranslateExceptionAndRaise(e, bucket_name=bucket.name)
        if not fields or 'encryption' in fields:
            try:
                keyname = bucket_uri.get_encryption_config(headers=headers).default_kms_key_name
                if keyname:
                    cloud_api_bucket.encryption = apitools_messages.Bucket.EncryptionValue()
                    cloud_api_bucket.encryption.defaultKmsKeyName = keyname
            except TRANSLATABLE_BOTO_EXCEPTIONS as e:
                self._TranslateExceptionAndRaise(e, bucket_name=bucket.name)
        if not fields or 'lifecycle' in fields:
            try:
                boto_lifecycle = bucket_uri.get_lifecycle_config(headers=headers)
                cloud_api_bucket.lifecycle = LifecycleTranslation.BotoLifecycleToMessage(boto_lifecycle)
            except TRANSLATABLE_BOTO_EXCEPTIONS as e:
                self._TranslateExceptionAndRaise(e, bucket_name=bucket.name)
        if not fields or 'logging' in fields:
            try:
                boto_logging = bucket_uri.get_logging_config(headers=headers)
                if boto_logging and 'Logging' in boto_logging:
                    logging_config = boto_logging['Logging']
                    log_object_prefix_present = 'LogObjectPrefix' in logging_config
                    log_bucket_present = 'LogBucket' in logging_config
                    if log_object_prefix_present or log_bucket_present:
                        cloud_api_bucket.logging = apitools_messages.Bucket.LoggingValue()
                        if log_object_prefix_present:
                            cloud_api_bucket.logging.logObjectPrefix = logging_config['LogObjectPrefix']
                        if log_bucket_present:
                            cloud_api_bucket.logging.logBucket = logging_config['LogBucket']
            except TRANSLATABLE_BOTO_EXCEPTIONS as e:
                self._TranslateExceptionAndRaise(e, bucket_name=bucket.name)
        if not fields or 'website' in fields:
            try:
                boto_website = bucket_uri.get_website_config(headers=headers)
                if boto_website and 'WebsiteConfiguration' in boto_website:
                    website_config = boto_website['WebsiteConfiguration']
                    main_page_suffix_present = 'MainPageSuffix' in website_config
                    not_found_page_present = 'NotFoundPage' in website_config
                    if main_page_suffix_present or not_found_page_present:
                        cloud_api_bucket.website = apitools_messages.Bucket.WebsiteValue()
                        if main_page_suffix_present:
                            cloud_api_bucket.website.mainPageSuffix = website_config['MainPageSuffix']
                        if not_found_page_present:
                            cloud_api_bucket.website.notFoundPage = website_config['NotFoundPage']
            except TRANSLATABLE_BOTO_EXCEPTIONS as e:
                self._TranslateExceptionAndRaise(e, bucket_name=bucket.name)
        if not fields or 'location' in fields:
            cloud_api_bucket.location = bucket_uri.get_location(headers=headers)
    if not fields or 'labels' in fields:
        try:
            try:
                boto_tags = bucket_uri.get_bucket().get_tags()
                cloud_api_bucket.labels = LabelTranslation.BotoTagsToMessage(boto_tags)
            except boto.exception.StorageResponseError as e:
                if not (self.provider == 's3' and e.status == 404):
                    raise
        except TRANSLATABLE_BOTO_EXCEPTIONS as e:
            self._TranslateExceptionAndRaise(e, bucket_name=bucket.name)
    if not fields or 'versioning' in fields:
        versioning = bucket_uri.get_versioning_config(headers=headers)
        if versioning:
            if self.provider == 's3' and 'Versioning' in versioning and (versioning['Versioning'] == 'Enabled'):
                cloud_api_bucket.versioning = apitools_messages.Bucket.VersioningValue(enabled=True)
            elif self.provider == 'gs':
                cloud_api_bucket.versioning = apitools_messages.Bucket.VersioningValue(enabled=True)
    return cloud_api_bucket