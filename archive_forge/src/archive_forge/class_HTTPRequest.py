from datetime import datetime
import errno
import os
import random
import re
import socket
import sys
import time
import xml.sax
import copy
from boto import auth
from boto import auth_handler
import boto
import boto.utils
import boto.handler
import boto.cacerts
from boto import config, UserAgent
from boto.compat import six, http_client, urlparse, quote, encodebytes
from boto.exception import AWSConnectionError
from boto.exception import BotoClientError
from boto.exception import BotoServerError
from boto.exception import PleaseRetryException
from boto.exception import S3ResponseError
from boto.provider import Provider
from boto.resultset import ResultSet
class HTTPRequest(object):

    def __init__(self, method, protocol, host, port, path, auth_path, params, headers, body):
        """Represents an HTTP request.

        :type method: string
        :param method: The HTTP method name, 'GET', 'POST', 'PUT' etc.

        :type protocol: string
        :param protocol: The http protocol used, 'http' or 'https'.

        :type host: string
        :param host: Host to which the request is addressed. eg. abc.com

        :type port: int
        :param port: port on which the request is being sent. Zero means unset,
            in which case default port will be chosen.

        :type path: string
        :param path: URL path that is being accessed.

        :type auth_path: string
        :param path: The part of the URL path used when creating the
            authentication string.

        :type params: dict
        :param params: HTTP url query parameters, with key as name of
            the param, and value as value of param.

        :type headers: dict
        :param headers: HTTP headers, with key as name of the header and value
            as value of header.

        :type body: string
        :param body: Body of the HTTP request. If not present, will be None or
            empty string ('').
        """
        self.method = method
        self.protocol = protocol
        self.host = host
        self.port = port
        self.path = path
        if auth_path is None:
            auth_path = path
        self.auth_path = auth_path
        self.params = params
        if headers and 'Transfer-Encoding' in headers and (headers['Transfer-Encoding'] == 'chunked') and (self.method != 'PUT'):
            self.headers = headers.copy()
            del self.headers['Transfer-Encoding']
        else:
            self.headers = headers
        self.body = body

    def __str__(self):
        return 'method:(%s) protocol:(%s) host(%s) port(%s) path(%s) params(%s) headers(%s) body(%s)' % (self.method, self.protocol, self.host, self.port, self.path, self.params, self.headers, self.body)

    def authorize(self, connection, **kwargs):
        if not getattr(self, '_headers_quoted', False):
            for key in self.headers:
                val = self.headers[key]
                if isinstance(val, six.text_type):
                    safe = '!"#$%&\'()*+,/:;<=>?@[\\]^`{|}~ '
                    self.headers[key] = quote(val.encode('utf-8'), safe)
            setattr(self, '_headers_quoted', True)
        self.headers['User-Agent'] = UserAgent
        connection._auth_handler.add_auth(self, **kwargs)
        if 'Content-Length' not in self.headers:
            if 'Transfer-Encoding' not in self.headers or self.headers['Transfer-Encoding'] != 'chunked':
                self.headers['Content-Length'] = str(len(self.body))