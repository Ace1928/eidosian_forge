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
def __ProcessHttpResponse(self, method_config, http_response, request):
    """Process the given http response."""
    if http_response.status_code not in (http_client.OK, http_client.CREATED, http_client.NO_CONTENT):
        raise exceptions.HttpError.FromResponse(http_response, method_config=method_config, request=request)
    if http_response.status_code == http_client.NO_CONTENT:
        http_response = http_wrapper.Response(info=http_response.info, content='{}', request_url=http_response.request_url)
    content = http_response.content
    if self._client.response_encoding and isinstance(content, bytes):
        content = content.decode(self._client.response_encoding)
    if self.__client.response_type_model == 'json':
        return content
    response_type = _LoadClass(method_config.response_type_name, self.__client.MESSAGES_MODULE)
    return self.__client.DeserializeMessage(response_type, content)