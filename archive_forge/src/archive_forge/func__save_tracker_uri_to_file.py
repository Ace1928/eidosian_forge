import errno
import os
import random
import re
import socket
import time
from hashlib import md5
import six.moves.http_client as httplib
from six.moves import urllib as urlparse
from boto import config, UserAgent
from boto.connection import AWSAuthConnection
from boto.exception import InvalidUriError
from boto.exception import ResumableTransferDisposition
from boto.exception import ResumableUploadException
from boto.s3.keyfile import KeyFile
def _save_tracker_uri_to_file(self):
    """
        Saves URI to tracker file if one was passed to constructor.
        """
    if not self.tracker_file_name:
        return
    f = None
    try:
        with os.fdopen(os.open(self.tracker_file_name, os.O_WRONLY | os.O_CREAT, 384), 'w') as f:
            f.write(self.tracker_uri)
    except IOError as e:
        raise ResumableUploadException("Couldn't write URI tracker file (%s): %s.\nThis can happenif you're using an incorrectly configured upload tool\n(e.g., gsutil configured to save tracker files to an unwritable directory)" % (self.tracker_file_name, e.strerror), ResumableTransferDisposition.ABORT)