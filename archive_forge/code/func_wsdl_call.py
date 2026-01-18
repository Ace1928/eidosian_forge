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
def wsdl_call(self, method, *args, **kwargs):
    """Pre and post process SOAP call, input and output parameters using WSDL"""
    return self.wsdl_call_with_args(method, args, kwargs)