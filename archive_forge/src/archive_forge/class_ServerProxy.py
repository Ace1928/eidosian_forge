import base64
import sys
import time
from datetime import datetime
from decimal import Decimal
import http.client
import urllib.parse
from xml.parsers import expat
import errno
from io import BytesIO
class ServerProxy:
    """uri [,options] -> a logical connection to an XML-RPC server

    uri is the connection point on the server, given as
    scheme://host/target.

    The standard implementation always supports the "http" scheme.  If
    SSL socket support is available (Python 2.0), it also supports
    "https".

    If the target part and the slash preceding it are both omitted,
    "/RPC2" is assumed.

    The following options can be given as keyword arguments:

        transport: a transport factory
        encoding: the request encoding (default is UTF-8)

    All 8-bit strings passed to the server proxy are assumed to use
    the given encoding.
    """

    def __init__(self, uri, transport=None, encoding=None, verbose=False, allow_none=False, use_datetime=False, use_builtin_types=False, *, headers=(), context=None):
        p = urllib.parse.urlsplit(uri)
        if p.scheme not in ('http', 'https'):
            raise OSError('unsupported XML-RPC protocol')
        self.__host = p.netloc
        self.__handler = urllib.parse.urlunsplit(['', '', *p[2:]])
        if not self.__handler:
            self.__handler = '/RPC2'
        if transport is None:
            if p.scheme == 'https':
                handler = SafeTransport
                extra_kwargs = {'context': context}
            else:
                handler = Transport
                extra_kwargs = {}
            transport = handler(use_datetime=use_datetime, use_builtin_types=use_builtin_types, headers=headers, **extra_kwargs)
        self.__transport = transport
        self.__encoding = encoding or 'utf-8'
        self.__verbose = verbose
        self.__allow_none = allow_none

    def __close(self):
        self.__transport.close()

    def __request(self, methodname, params):
        request = dumps(params, methodname, encoding=self.__encoding, allow_none=self.__allow_none).encode(self.__encoding, 'xmlcharrefreplace')
        response = self.__transport.request(self.__host, self.__handler, request, verbose=self.__verbose)
        if len(response) == 1:
            response = response[0]
        return response

    def __repr__(self):
        return '<%s for %s%s>' % (self.__class__.__name__, self.__host, self.__handler)

    def __getattr__(self, name):
        return _Method(self.__request, name)

    def __call__(self, attr):
        """A workaround to get special attributes on the ServerProxy
           without interfering with the magic __getattr__
        """
        if attr == 'close':
            return self.__close
        elif attr == 'transport':
            return self.__transport
        raise AttributeError('Attribute %r not found' % (attr,))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.__close()