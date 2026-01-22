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
class BaseDriver:
    """
    Base driver class from which other classes can inherit from.
    """
    connectionCls = ConnectionKey

    def __init__(self, key, secret=None, secure=True, host=None, port=None, api_version=None, region=None, **kwargs):
        """
        :param    key:    API key or username to be used (required)
        :type     key:    ``str``

        :param    secret: Secret password to be used (required)
        :type     secret: ``str``

        :param    secure: Whether to use HTTPS or HTTP. Note: Some providers
                            only support HTTPS, and it is on by default.
        :type     secure: ``bool``

        :param    host: Override hostname used for connections.
        :type     host: ``str``

        :param    port: Override port used for connections.
        :type     port: ``int``

        :param    api_version: Optional API version. Only used by drivers
                                 which support multiple API versions.
        :type     api_version: ``str``

        :param region: Optional driver region. Only used by drivers which
                       support multiple regions.
        :type region: ``str``

        :rtype: ``None``
        """
        self.key = key
        self.secret = secret
        self.secure = secure
        self.api_version = api_version
        self.region = region
        conn_kwargs = self._ex_connection_class_kwargs()
        conn_kwargs.update({'timeout': kwargs.pop('timeout', None), 'retry_delay': kwargs.pop('retry_delay', None), 'backoff': kwargs.pop('backoff', None), 'proxy_url': kwargs.pop('proxy_url', None)})
        args = [self.key]
        if self.secret is not None:
            args.append(self.secret)
        args.append(secure)
        host = conn_kwargs.pop('host', None) or host
        if host is not None:
            args.append(host)
        if port is not None:
            args.append(port)
        self.connection = self.connectionCls(*args, **conn_kwargs)
        self.connection.driver = self
        self.connection.connect()

    def _ex_connection_class_kwargs(self):
        """
        Return extra connection keyword arguments which are passed to the
        Connection class constructor.
        """
        return {}