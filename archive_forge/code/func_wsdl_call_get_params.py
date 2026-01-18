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
def wsdl_call_get_params(self, method, input, args, kwargs):
    """Build params from input and args/kwargs"""
    params = inputname = inputargs = None
    all_args = {}
    if input:
        inputname = list(input.keys())[0]
        inputargs = input[inputname]
    if input and args:
        d = {}
        for idx, arg in enumerate(args):
            key = list(inputargs.keys())[idx]
            if isinstance(arg, dict):
                if key not in arg:
                    raise KeyError('Unhandled key %s. use client.help(method)' % key)
                d[key] = arg[key]
            else:
                d[key] = arg
        all_args.update({inputname: d})
    if input and (kwargs or all_args):
        if kwargs:
            all_args.update({inputname: kwargs})
        valid, errors, warnings = self.wsdl_validate_params(input, all_args)
        if not valid:
            raise ValueError('Invalid Args Structure. Errors: %s' % errors)
        tree = sort_dict(input, all_args)
        root = list(tree.values())[0]
        params = []
        for k, v in root.items():
            root_ns = root.namespaces[k]
            if not root.references[k] and isinstance(v, Struct):
                v.namespaces[None] = root_ns
            params.append((k, v))
        if self.__soap_server in ('axis',):
            method = method
        else:
            method = inputname
    else:
        params = kwargs and kwargs.items()
    return (method, params)