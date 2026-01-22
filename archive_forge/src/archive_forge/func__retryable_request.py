import os
import ssl
import copy
import json
import time
import socket
import binascii
from typing import Any, Dict, Type, Union, Optional
import libcloud
from libcloud.http import LibcloudConnection, HttpLibResponseProxy
from libcloud.utils.py3 import ET, httplib, urlparse, urlencode
from libcloud.utils.misc import lowercase_keys
from libcloud.utils.retry import Retry
from libcloud.common.types import LibcloudError, MalformedResponseError
from libcloud.common.exceptions import exception_from_message
def _retryable_request(self, url: str, data: bytes, headers: Dict[str, Any], method: str, raw: bool, stream: bool) -> Union[RawResponse, Response]:
    try:
        assert self.connection is not None
        if raw:
            self.connection.prepared_request(method=method, url=url, body=data, headers=headers, raw=raw, stream=stream)
        else:
            self.connection.request(method=method, url=url, body=data, headers=headers, stream=stream)
    except socket.gaierror as e:
        message = str(e)
        errno = getattr(e, 'errno', None)
        if errno == -5:
            class_name = self.__class__.__name__
            msg = '%s. Perhaps "host" Connection class attribute (%s.connection) is set to an invalid, non-hostname value (%s)?' % (message, class_name, self.host)
            raise socket.gaierror(msg)
        self.reset_context()
        raise e
    except ssl.SSLError as e:
        self.reset_context()
        raise ssl.SSLError(str(e))
    if raw:
        responseCls = self.rawResponseCls
        kwargs = {'connection': self, 'response': self.connection.getresponse()}
    else:
        responseCls = self.responseCls
        kwargs = {'connection': self, 'response': self.connection.getresponse()}
    try:
        response = responseCls(**kwargs)
    finally:
        self.reset_context()
    return response