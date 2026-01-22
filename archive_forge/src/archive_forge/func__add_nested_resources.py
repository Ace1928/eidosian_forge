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
def _add_nested_resources(self, resourceDesc, rootDesc, schema):
    if 'resources' in resourceDesc:

        def createResourceMethod(methodName, methodDesc):
            """Create a method on the Resource to access a nested Resource.

        Args:
          methodName: string, name of the method to use.
          methodDesc: object, fragment of deserialized discovery document that
            describes the method.
        """
            methodName = fix_method_name(methodName)

            def methodResource(self):
                return Resource(http=self._http, baseUrl=self._baseUrl, model=self._model, developerKey=self._developerKey, requestBuilder=self._requestBuilder, resourceDesc=methodDesc, rootDesc=rootDesc, schema=schema)
            setattr(methodResource, '__doc__', 'A collection resource.')
            setattr(methodResource, '__is_resource__', True)
            return (methodName, methodResource)
        for methodName, methodDesc in six.iteritems(resourceDesc['resources']):
            fixedMethodName, method = createResourceMethod(methodName, methodDesc)
            self._set_dynamic_attr(fixedMethodName, method.__get__(self, self.__class__))