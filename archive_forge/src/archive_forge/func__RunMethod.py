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
def _RunMethod(self, method_config, request, global_params=None, upload=None, upload_config=None, download=None):
    """Call this method with request."""
    if upload is not None and download is not None:
        raise exceptions.NotYetImplementedError('Cannot yet use both upload and download at once')
    http_request = self.PrepareHttpRequest(method_config, request, global_params, upload, upload_config, download)
    if download is not None:
        download.InitializeDownload(http_request, client=self.client)
        return
    http_response = None
    if upload is not None:
        http_response = upload.InitializeUpload(http_request, client=self.client)
    if http_response is None:
        http = self.__client.http
        if upload and upload.bytes_http:
            http = upload.bytes_http
        opts = {'retries': self.__client.num_retries, 'max_retry_wait': self.__client.max_retry_wait}
        if self.__client.check_response_func:
            opts['check_response_func'] = self.__client.check_response_func
        if self.__client.retry_func:
            opts['retry_func'] = self.__client.retry_func
        http_response = http_wrapper.MakeRequest(http, http_request, **opts)
    return self.ProcessHttpResponse(method_config, http_response, request)