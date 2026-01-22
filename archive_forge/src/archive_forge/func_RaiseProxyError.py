from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import abc
import contextlib
import select
import socket
import sys
import webbrowser
import wsgiref
from google_auth_oauthlib import flow as google_auth_flow
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as c_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import requests
from googlecloudsdk.core.console import console_attr
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import pkg_resources
from oauthlib.oauth2.rfc6749 import errors as rfc6749_errors
from requests import exceptions as requests_exceptions
import six
from six.moves import input  # pylint: disable=redefined-builtin
from six.moves.urllib import parse
def RaiseProxyError(source_exc):
    six.raise_from(AuthRequestFailedError('Could not reach the login server. A potential cause of this could be because you are behind a proxy. Please set the environment variables HTTPS_PROXY and HTTP_PROXY to the address of the proxy in the format "protocol://address:port" (without quotes) and try again.\nExample: HTTPS_PROXY=http://192.168.0.1:8080'), source_exc)