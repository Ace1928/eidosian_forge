from __future__ import absolute_import
import six
from six.moves import zip
from six import BytesIO
from six.moves import http_client
from six.moves.urllib.parse import urlencode, urlparse, urljoin, urlunparse, parse_qsl
import copy
from collections import OrderedDict
from email.mime.multipart import MIMEMultipart
from email.mime.nonmultipart import MIMENonMultipart
import json
import keyword
import logging
import mimetypes
import os
import re
import httplib2
import uritemplate
import google.api_core.client_options
from google.auth.transport import mtls
from google.auth.exceptions import MutualTLSChannelError
from googleapiclient import _auth
from googleapiclient import mimeparse
from googleapiclient.errors import HttpError
from googleapiclient.errors import InvalidJsonError
from googleapiclient.errors import MediaUploadSizeError
from googleapiclient.errors import UnacceptableMimeTypeError
from googleapiclient.errors import UnknownApiNameOrVersion
from googleapiclient.errors import UnknownFileType
from googleapiclient.http import build_http
from googleapiclient.http import BatchHttpRequest
from googleapiclient.http import HttpMock
from googleapiclient.http import HttpMockSequence
from googleapiclient.http import HttpRequest
from googleapiclient.http import MediaFileUpload
from googleapiclient.http import MediaUpload
from googleapiclient.model import JsonModel
from googleapiclient.model import MediaModel
from googleapiclient.model import RawModel
from googleapiclient.schema import Schemas
from googleapiclient._helpers import _add_query_parameter
from googleapiclient._helpers import positional
class ResourceMethodParameters(object):
    """Represents the parameters associated with a method.

  Attributes:
    argmap: Map from method parameter name (string) to query parameter name
        (string).
    required_params: List of required parameters (represented by parameter
        name as string).
    repeated_params: List of repeated parameters (represented by parameter
        name as string).
    pattern_params: Map from method parameter name (string) to regular
        expression (as a string). If the pattern is set for a parameter, the
        value for that parameter must match the regular expression.
    query_params: List of parameters (represented by parameter name as string)
        that will be used in the query string.
    path_params: Set of parameters (represented by parameter name as string)
        that will be used in the base URL path.
    param_types: Map from method parameter name (string) to parameter type. Type
        can be any valid JSON schema type; valid values are 'any', 'array',
        'boolean', 'integer', 'number', 'object', or 'string'. Reference:
        http://tools.ietf.org/html/draft-zyp-json-schema-03#section-5.1
    enum_params: Map from method parameter name (string) to list of strings,
       where each list of strings is the list of acceptable enum values.
  """

    def __init__(self, method_desc):
        """Constructor for ResourceMethodParameters.

    Sets default values and defers to set_parameters to populate.

    Args:
      method_desc: Dictionary with metadata describing an API method. Value
          comes from the dictionary of methods stored in the 'methods' key in
          the deserialized discovery document.
    """
        self.argmap = {}
        self.required_params = []
        self.repeated_params = []
        self.pattern_params = {}
        self.query_params = []
        self.path_params = set()
        self.param_types = {}
        self.enum_params = {}
        self.set_parameters(method_desc)

    def set_parameters(self, method_desc):
        """Populates maps and lists based on method description.

    Iterates through each parameter for the method and parses the values from
    the parameter dictionary.

    Args:
      method_desc: Dictionary with metadata describing an API method. Value
          comes from the dictionary of methods stored in the 'methods' key in
          the deserialized discovery document.
    """
        parameters = method_desc.get('parameters', {})
        sorted_parameters = OrderedDict(sorted(parameters.items()))
        for arg, desc in six.iteritems(sorted_parameters):
            param = key2param(arg)
            self.argmap[param] = arg
            if desc.get('pattern'):
                self.pattern_params[param] = desc['pattern']
            if desc.get('enum'):
                self.enum_params[param] = desc['enum']
            if desc.get('required'):
                self.required_params.append(param)
            if desc.get('repeated'):
                self.repeated_params.append(param)
            if desc.get('location') == 'query':
                self.query_params.append(param)
            if desc.get('location') == 'path':
                self.path_params.add(param)
            self.param_types[param] = desc.get('type', 'string')
        for match in URITEMPLATE.finditer(method_desc['path']):
            for namematch in VARNAME.finditer(match.group(0)):
                name = key2param(namematch.group(0))
                self.path_params.add(name)
                if name in self.query_params:
                    self.query_params.remove(name)