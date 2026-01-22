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
class ApiMethodInfo(messages.Message):
    """Configuration info for an API method.

    All fields are strings unless noted otherwise.

    Fields:
      relative_path: Relative path for this method.
      flat_path: Expanded version (if any) of relative_path.
      method_id: ID for this method.
      http_method: HTTP verb to use for this method.
      path_params: (repeated) path parameters for this method.
      query_params: (repeated) query parameters for this method.
      ordered_params: (repeated) ordered list of parameters for
          this method.
      description: description of this method.
      request_type_name: name of the request type.
      response_type_name: name of the response type.
      request_field: if not null, the field to pass as the body
          of this POST request. may also be the REQUEST_IS_BODY
          value below to indicate the whole message is the body.
      upload_config: (ApiUploadInfo) Information about the upload
          configuration supported by this method.
      supports_download: (boolean) If True, this method supports
          downloading the request via the `alt=media` query
          parameter.
    """
    relative_path = messages.StringField(1)
    flat_path = messages.StringField(2)
    method_id = messages.StringField(3)
    http_method = messages.StringField(4)
    path_params = messages.StringField(5, repeated=True)
    query_params = messages.StringField(6, repeated=True)
    ordered_params = messages.StringField(7, repeated=True)
    description = messages.StringField(8)
    request_type_name = messages.StringField(9)
    response_type_name = messages.StringField(10)
    request_field = messages.StringField(11, default='')
    upload_config = messages.MessageField(ApiUploadInfo, 12)
    supports_download = messages.BooleanField(13, default=False)