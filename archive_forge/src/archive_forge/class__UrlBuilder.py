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
class _UrlBuilder(object):
    """Convenient container for url data."""

    def __init__(self, base_url, relative_path=None, query_params=None):
        components = urllib.parse.urlsplit(_urljoin(base_url, relative_path or ''))
        if components.fragment:
            raise exceptions.ConfigurationValueError('Unexpected url fragment: %s' % components.fragment)
        self.query_params = urllib.parse.parse_qs(components.query or '')
        if query_params is not None:
            self.query_params.update(query_params)
        self.__scheme = components.scheme
        self.__netloc = components.netloc
        self.relative_path = components.path or ''

    @classmethod
    def FromUrl(cls, url):
        urlparts = urllib.parse.urlsplit(url)
        query_params = urllib.parse.parse_qs(urlparts.query)
        base_url = urllib.parse.urlunsplit((urlparts.scheme, urlparts.netloc, '', None, None))
        relative_path = urlparts.path or ''
        return cls(base_url, relative_path=relative_path, query_params=query_params)

    @property
    def base_url(self):
        return urllib.parse.urlunsplit((self.__scheme, self.__netloc, '', '', ''))

    @base_url.setter
    def base_url(self, value):
        components = urllib.parse.urlsplit(value)
        if components.path or components.query or components.fragment:
            raise exceptions.ConfigurationValueError('Invalid base url: %s' % value)
        self.__scheme = components.scheme
        self.__netloc = components.netloc

    @property
    def query(self):
        return urllib.parse.urlencode(self.query_params, True)

    @property
    def url(self):
        if '{' in self.relative_path or '}' in self.relative_path:
            raise exceptions.ConfigurationValueError('Cannot create url with relative path %s' % self.relative_path)
        return urllib.parse.urlunsplit((self.__scheme, self.__netloc, self.relative_path, self.query, ''))