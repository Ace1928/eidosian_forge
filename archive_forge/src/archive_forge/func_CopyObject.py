from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
from __future__ import unicode_literals
from contextlib import contextmanager
import functools
from six.moves import http_client
import json
import logging
import os
import socket
import ssl
import time
import traceback
import six
from apitools.base.py import encoding
from apitools.base.py import exceptions as apitools_exceptions
from apitools.base.py import http_wrapper as apitools_http_wrapper
from apitools.base.py import transfer as apitools_transfer
from apitools.base.py.util import CalculateWaitForRetry
from boto import config
import httplib2
import oauth2client
from gslib import context_config
from gslib.cloud_api import AccessDeniedException
from gslib.cloud_api import ArgumentException
from gslib.cloud_api import BadRequestException
from gslib.cloud_api import CloudApi
from gslib.cloud_api import EncryptionException
from gslib.cloud_api import NotEmptyException
from gslib.cloud_api import NotFoundException
from gslib.cloud_api import PreconditionException
from gslib.cloud_api import Preconditions
from gslib.cloud_api import PublishPermissionDeniedException
from gslib.cloud_api import ResumableDownloadException
from gslib.cloud_api import ResumableUploadAbortException
from gslib.cloud_api import ResumableUploadException
from gslib.cloud_api import ResumableUploadStartOverException
from gslib.cloud_api import ServiceException
from gslib.exception import CommandException
from gslib.gcs_json_credentials import SetUpJsonCredentialsAndCache
from gslib.gcs_json_media import BytesTransferredContainer
from gslib.gcs_json_media import DownloadCallbackConnectionClassFactory
from gslib.gcs_json_media import HttpWithDownloadStream
from gslib.gcs_json_media import HttpWithNoRetries
from gslib.gcs_json_media import UploadCallbackConnectionClassFactory
from gslib.gcs_json_media import WrapDownloadHttpRequest
from gslib.gcs_json_media import WrapUploadHttpRequest
from gslib.impersonation_credentials import ImpersonationCredentials
from gslib.no_op_credentials import NoOpCredentials
from gslib.progress_callback import ProgressCallbackWithTimeout
from gslib.project_id import PopulateProjectId
from gslib.third_party.storage_apitools import storage_v1_client as apitools_client
from gslib.third_party.storage_apitools import storage_v1_messages as apitools_messages
from gslib.tracker_file import DeleteTrackerFile
from gslib.tracker_file import GetRewriteTrackerFilePath
from gslib.tracker_file import HashRewriteParameters
from gslib.tracker_file import ReadRewriteTrackerFile
from gslib.tracker_file import WriteRewriteTrackerFile
from gslib.utils.boto_util import GetCertsFile
from gslib.utils.boto_util import GetGcsJsonApiVersion
from gslib.utils.boto_util import GetJsonResumableChunkSize
from gslib.utils.boto_util import GetMaxRetryDelay
from gslib.utils.boto_util import GetNewHttp
from gslib.utils.boto_util import GetNumRetries
from gslib.utils.boto_util import JsonResumableChunkSizeDefined
from gslib.utils.cloud_api_helper import ListToGetFields
from gslib.utils.cloud_api_helper import ValidateDstObjectMetadata
from gslib.utils.constants import IAM_POLICY_VERSION
from gslib.utils.constants import NUM_OBJECTS_PER_LIST_PAGE
from gslib.utils.constants import REQUEST_REASON_ENV_VAR
from gslib.utils.constants import REQUEST_REASON_HEADER_KEY
from gslib.utils.constants import UTF8
from gslib.utils.encryption_helper import Base64Sha256FromBase64EncryptionKey
from gslib.utils.encryption_helper import CryptoKeyType
from gslib.utils.encryption_helper import CryptoKeyWrapperFromKey
from gslib.utils.encryption_helper import FindMatchingCSEKInBotoConfig
from gslib.utils.metadata_util import AddAcceptEncodingGzipIfNeeded
from gslib.utils.retry_util import LogAndHandleRetries
from gslib.utils.signurl_helper import CreatePayload, GetFinalUrl
from gslib.utils.text_util import GetPrintableExceptionString
from gslib.utils.translation_helper import CreateBucketNotFoundException
from gslib.utils.translation_helper import CreateNotFoundExceptionForObjectWrite
from gslib.utils.translation_helper import CreateObjectNotFoundException
from gslib.utils.translation_helper import DEFAULT_CONTENT_TYPE
from gslib.utils.translation_helper import PRIVATE_DEFAULT_OBJ_ACL
from gslib.utils.translation_helper import REMOVE_CORS_CONFIG
from oauth2client.service_account import ServiceAccountCredentials
def CopyObject(self, src_obj_metadata, dst_obj_metadata, src_generation=None, canned_acl=None, preconditions=None, progress_callback=None, max_bytes_per_call=None, encryption_tuple=None, decryption_tuple=None, provider=None, fields=None):
    """See CloudApi class for function doc strings."""
    ValidateDstObjectMetadata(dst_obj_metadata)
    predefined_acl = None
    if canned_acl:
        predefined_acl = apitools_messages.StorageObjectsRewriteRequest.DestinationPredefinedAclValueValuesEnum(self._ObjectCannedAclToPredefinedAcl(canned_acl))
    if src_generation:
        src_generation = long(src_generation)
    if not preconditions:
        preconditions = Preconditions()
    projection = apitools_messages.StorageObjectsRewriteRequest.ProjectionValueValuesEnum.noAcl
    if self._FieldsContainsAclField(fields):
        projection = apitools_messages.StorageObjectsRewriteRequest.ProjectionValueValuesEnum.full
    global_params = apitools_messages.StandardQueryParameters()
    if fields:
        new_fields = set(['done', 'objectSize', 'rewriteToken', 'totalBytesRewritten'])
        for field in fields:
            new_fields.add('resource/' + field)
        global_params.fields = ','.join(set(new_fields))
    dec_key_sha256 = None
    if decryption_tuple and decryption_tuple.crypto_type == CryptoKeyType.CSEK:
        dec_key_sha256 = decryption_tuple.crypto_key_sha256
    enc_key_sha256 = None
    if encryption_tuple:
        if encryption_tuple.crypto_type == CryptoKeyType.CSEK:
            enc_key_sha256 = encryption_tuple.crypto_key_sha256
        elif encryption_tuple.crypto_type == CryptoKeyType.CMEK:
            dst_obj_metadata.kmsKeyName = encryption_tuple.crypto_key
    tracker_file_name = GetRewriteTrackerFilePath(src_obj_metadata.bucket, src_obj_metadata.name, dst_obj_metadata.bucket, dst_obj_metadata.name, 'JSON')
    rewrite_params_hash = HashRewriteParameters(src_obj_metadata, dst_obj_metadata, projection, src_generation=src_generation, gen_match=preconditions.gen_match, meta_gen_match=preconditions.meta_gen_match, canned_acl=predefined_acl, max_bytes_per_call=max_bytes_per_call, src_dec_key_sha256=dec_key_sha256, dst_enc_key_sha256=enc_key_sha256, fields=global_params.fields)
    resume_rewrite_token = ReadRewriteTrackerFile(tracker_file_name, rewrite_params_hash)
    crypto_headers = self._RewriteCryptoHeadersFromTuples(decryption_tuple=decryption_tuple, encryption_tuple=encryption_tuple)
    progress_cb_with_timeout = None
    try:
        last_bytes_written = long(0)
        while True:
            with self._ApitoolsRequestHeaders(crypto_headers):
                apitools_request = apitools_messages.StorageObjectsRewriteRequest(sourceBucket=src_obj_metadata.bucket, sourceObject=src_obj_metadata.name, destinationBucket=dst_obj_metadata.bucket, destinationKmsKeyName=dst_obj_metadata.kmsKeyName, destinationObject=dst_obj_metadata.name, projection=projection, object=dst_obj_metadata, sourceGeneration=src_generation, ifGenerationMatch=preconditions.gen_match, ifMetagenerationMatch=preconditions.meta_gen_match, destinationPredefinedAcl=predefined_acl, rewriteToken=resume_rewrite_token, maxBytesRewrittenPerCall=max_bytes_per_call, userProject=self.user_project)
                rewrite_response = self.api_client.objects.Rewrite(apitools_request, global_params=global_params)
            bytes_written = long(rewrite_response.totalBytesRewritten)
            if progress_callback and (not progress_cb_with_timeout):
                progress_cb_with_timeout = ProgressCallbackWithTimeout(long(rewrite_response.objectSize), progress_callback)
            if progress_cb_with_timeout:
                progress_cb_with_timeout.Progress(bytes_written - last_bytes_written)
            if rewrite_response.done:
                break
            elif not resume_rewrite_token:
                resume_rewrite_token = rewrite_response.rewriteToken
                WriteRewriteTrackerFile(tracker_file_name, rewrite_params_hash, rewrite_response.rewriteToken)
            last_bytes_written = bytes_written
        DeleteTrackerFile(tracker_file_name)
        return rewrite_response.resource
    except TRANSLATABLE_APITOOLS_EXCEPTIONS as e:
        not_found_exception = CreateNotFoundExceptionForObjectWrite(self.provider, dst_obj_metadata.bucket, src_provider=self.provider, src_bucket_name=src_obj_metadata.bucket, src_object_name=src_obj_metadata.name, src_generation=src_generation)
        self._TranslateExceptionAndRaise(e, bucket_name=dst_obj_metadata.bucket, object_name=dst_obj_metadata.name, not_found_exception=not_found_exception)