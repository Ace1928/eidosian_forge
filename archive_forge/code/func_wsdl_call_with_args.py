from __future__ import unicode_literals
import sys
import copy
import hashlib
import logging
import os
import tempfile
import warnings
from . import __author__, __copyright__, __license__, __version__, TIMEOUT
from .simplexml import SimpleXMLElement, TYPE_MAP, REVERSE_TYPE_MAP, Struct
from .transport import get_http_wrapper, set_http_wrapper, get_Http
from .helpers import Alias, fetch, sort_dict, make_key, process_element, \
from .wsse import UsernameToken
def wsdl_call_with_args(self, method, args, kwargs):
    """Pre and post process SOAP call, input and output parameters using WSDL"""
    soap_uri = soap_namespaces[self.__soap_ns]
    operation = self.get_operation(method)
    input = operation['input']
    output = operation['output']
    header = operation.get('header')
    if 'action' in operation:
        self.action = operation['action']
    if 'namespace' in operation:
        self.namespace = operation['namespace'] or ''
        self.qualified = operation['qualified']
    if header:
        self.__call_headers = sort_dict(header, self.__headers)
    method, params = self.wsdl_call_get_params(method, input, args, kwargs)
    response = self.call(method, *params)
    resp = response('Body', ns=soap_uri).children().unmarshall(output, strict=self.strict)
    return resp and list(resp.values())[0]