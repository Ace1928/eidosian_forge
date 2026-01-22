import base64
import contextlib
import datetime
import logging
import pprint
import six
from six.moves import http_client
from six.moves import urllib
from apitools.base.protorpclite import message_types
from apitools.base.protorpclite import messages
from apitools.base.py import encoding
from apitools.base.py import exceptions
from apitools.base.py import http_wrapper
from apitools.base.py import util
class ApiUploadInfo(messages.Message):
    """Media upload information for a method.

    Fields:
      accept: (repeated) MIME Media Ranges for acceptable media uploads
          to this method.
      max_size: (integer) Maximum size of a media upload, such as 3MB
          or 1TB (converted to an integer).
      resumable_path: Path to use for resumable uploads.
      resumable_multipart: (boolean) Whether or not the resumable endpoint
          supports multipart uploads.
      simple_path: Path to use for simple uploads.
      simple_multipart: (boolean) Whether or not the simple endpoint
          supports multipart uploads.
    """
    accept = messages.StringField(1, repeated=True)
    max_size = messages.IntegerField(2)
    resumable_path = messages.StringField(3)
    resumable_multipart = messages.BooleanField(4)
    simple_path = messages.StringField(5)
    simple_multipart = messages.BooleanField(6)