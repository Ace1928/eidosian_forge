from __future__ import absolute_import
import datetime
import errno
from hashlib import sha1
import json
import logging
import os
import socket
import tempfile
import threading
import boto
import httplib2
import oauth2client.client
import oauth2client.service_account
from google_reauth import reauth_creds
import retry_decorator.retry_decorator
import six
from six import BytesIO
from six.moves import urllib
def CreateHttpRequest(self):
    return httplib2.Http(ca_certs=self.ca_certs_file, disable_ssl_certificate_validation=self.disable_ssl_certificate_validation, proxy_info=self._proxy_info)