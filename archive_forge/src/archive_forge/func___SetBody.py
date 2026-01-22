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
def __SetBody(self, http_request, method_config, request, upload):
    """Fill in the body on http_request."""
    if not method_config.request_field:
        return
    request_type = _LoadClass(method_config.request_type_name, self.__client.MESSAGES_MODULE)
    if method_config.request_field == REQUEST_IS_BODY:
        body_value = request
        body_type = request_type
    else:
        body_value = getattr(request, method_config.request_field)
        body_field = request_type.field_by_name(method_config.request_field)
        util.Typecheck(body_field, messages.MessageField)
        body_type = body_field.type
    body_value = body_value or body_type()
    if upload and (not body_value):
        return
    util.Typecheck(body_value, body_type)
    http_request.headers['content-type'] = 'application/json'
    http_request.body = self.__client.SerializeMessage(body_value)