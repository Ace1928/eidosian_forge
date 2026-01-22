import xml.sax
import base64
import time
from boto.compat import six, urllib
from boto.auth import detect_potential_s3sigv4
import boto.utils
from boto.connection import AWSAuthConnection
from boto import handler
from boto.s3.bucket import Bucket
from boto.s3.key import Key
from boto.resultset import ResultSet
from boto.exception import BotoClientError, S3ResponseError
from boto.utils import get_utf8able_str
class _CallingFormat(object):

    def get_bucket_server(self, server, bucket):
        return ''

    def build_url_base(self, connection, protocol, server, bucket, key=''):
        url_base = '%s://' % six.ensure_text(protocol)
        url_base += self.build_host(server, bucket)
        url_base += connection.get_path(self.build_path_base(bucket, key))
        return url_base

    def build_host(self, server, bucket):
        if bucket == '':
            return server
        else:
            return self.get_bucket_server(server, bucket)

    def build_auth_path(self, bucket, key=u''):
        bucket = six.ensure_text(bucket, encoding='utf-8')
        key = get_utf8able_str(key)
        path = u''
        if bucket != u'':
            path = u'/' + bucket
        return path + '/%s' % urllib.parse.quote(key)

    def build_path_base(self, bucket, key=''):
        key = get_utf8able_str(key)
        return '/%s' % urllib.parse.quote(key)