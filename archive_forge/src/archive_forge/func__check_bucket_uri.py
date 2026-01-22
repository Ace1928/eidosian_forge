import boto
import os
import sys
import textwrap
from boto.s3.deletemarker import DeleteMarker
from boto.exception import BotoClientError
from boto.exception import InvalidUriError
def _check_bucket_uri(self, function_name):
    if issubclass(type(self), BucketStorageUri) and (not self.bucket_name):
        raise InvalidUriError('%s on bucket-less URI (%s)' % (function_name, self.uri))