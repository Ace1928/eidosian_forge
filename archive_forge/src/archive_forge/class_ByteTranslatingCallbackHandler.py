import errno
import os
import re
import socket
import time
import six.moves.http_client as httplib
import boto
from boto import config, storage_uri_for_key
from boto.connection import AWSAuthConnection
from boto.exception import ResumableDownloadException
from boto.exception import ResumableTransferDisposition
from boto.s3.keyfile import KeyFile
from boto.gs.key import Key as GSKey
class ByteTranslatingCallbackHandler(object):
    """
    Proxy class that translates progress callbacks made by
    boto.s3.Key.get_file(), taking into account that we're resuming
    a download.
    """

    def __init__(self, proxied_cb, download_start_point):
        self.proxied_cb = proxied_cb
        self.download_start_point = download_start_point

    def call(self, total_bytes_uploaded, total_size):
        self.proxied_cb(self.download_start_point + total_bytes_uploaded, total_size)