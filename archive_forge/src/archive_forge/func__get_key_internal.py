from __future__ import division
import boto
from boto import handler
from boto.resultset import ResultSet
from boto.exception import BotoClientError
from boto.s3.acl import Policy, CannedACLStrings, Grant
from boto.s3.key import Key
from boto.s3.prefix import Prefix
from boto.s3.deletemarker import DeleteMarker
from boto.s3.multipart import MultiPartUpload
from boto.s3.multipart import CompleteMultiPartUpload
from boto.s3.multidelete import MultiDeleteResult
from boto.s3.multidelete import Error
from boto.s3.bucketlistresultset import BucketListResultSet
from boto.s3.bucketlistresultset import VersionedBucketListResultSet
from boto.s3.bucketlistresultset import MultiPartUploadListResultSet
from boto.s3.lifecycle import Lifecycle
from boto.s3.tagging import Tags
from boto.s3.cors import CORSConfiguration
from boto.s3.bucketlogging import BucketLogging
from boto.s3 import website
import boto.jsonresponse
import boto.utils
import xml.sax
import xml.sax.saxutils
import re
import base64
from collections import defaultdict
from boto.compat import BytesIO, six, StringIO, urllib
from boto.utils import get_utf8able_str
def _get_key_internal(self, key_name, headers, query_args_l):
    query_args = '&'.join(query_args_l) or None
    response = self.connection.make_request('HEAD', self.name, key_name, headers=headers, query_args=query_args)
    response.read()
    if response.status // 100 == 2:
        k = self.key_class(self)
        provider = self.connection.provider
        k.metadata = boto.utils.get_aws_metadata(response.msg, provider)
        for field in Key.base_fields:
            k.__dict__[field.lower().replace('-', '_')] = response.getheader(field)
        clen = response.getheader('content-length')
        if clen:
            k.size = int(response.getheader('content-length'))
        else:
            k.size = 0
        k.name = key_name
        k.handle_version_headers(response)
        k.handle_encryption_headers(response)
        k.handle_restore_headers(response)
        k.handle_storage_class_header(response)
        k.handle_addl_headers(response.getheaders())
        return (k, response)
    elif response.status == 404:
        return (None, response)
    else:
        raise self.connection.provider.storage_response_error(response.status, response.reason, '')