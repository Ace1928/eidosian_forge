from datetime import datetime
import errno
import os
import random
import re
import socket
import sys
import time
import xml.sax
import copy
from boto import auth
from boto import auth_handler
import boto
import boto.utils
import boto.handler
import boto.cacerts
from boto import config, UserAgent
from boto.compat import six, http_client, urlparse, quote, encodebytes
from boto.exception import AWSConnectionError
from boto.exception import BotoClientError
from boto.exception import BotoServerError
from boto.exception import PleaseRetryException
from boto.exception import S3ResponseError
from boto.provider import Provider
from boto.resultset import ResultSet
def _change_s3_host_from_error(self, request, err, get_header=None):
    new_endpoint = self._get_correct_s3_endpoint_from_response(request, err, get_header)
    if not new_endpoint:
        return None
    msg = 'This request was sent to %s, ' % request.host
    msg += 'when it should have been sent to %s. ' % new_endpoint
    request.host = new_endpoint
    new_host = self._get_s3_host(new_endpoint)
    if new_host and new_host != self.host:
        msg += 'This error may have arisen because your S3 host, '
        msg += 'currently %s, is configured incorrectly. ' % self.host
        msg += 'Please change your configuration to use %s ' % new_host
        msg += 'to avoid multiple unnecessary redirects '
        msg += 'and signing attempts.'
        self.host = new_host
    boto.log.debug(msg)
    return request