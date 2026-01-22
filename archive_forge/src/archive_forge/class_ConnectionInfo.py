from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import abc
import base64
import contextlib
import os
import re
import ssl
import sys
import tempfile
from googlecloudsdk.api_lib.run import gke
from googlecloudsdk.api_lib.run import global_methods
from googlecloudsdk.api_lib.util import apis
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.run import exceptions as serverless_exceptions
from googlecloudsdk.command_lib.run import flags
from googlecloudsdk.command_lib.run import platforms
from googlecloudsdk.core import properties
from googlecloudsdk.core import transport
from googlecloudsdk.core.util import files
import requests
import six
from six.moves.urllib import parse as urlparse
class ConnectionInfo(six.with_metaclass(abc.ABCMeta)):
    """Information useful in constructing an API client."""

    def __init__(self, api_name, version):
        """Initialize a connection context.

    Args:
      api_name: str, api name to use for making requests.
      version: str, api version to use for making requests.
    """
        self.endpoint = None
        self.ca_certs = None
        self.region = None
        self._cm = None
        self._api_name = api_name
        self._version = version

    @property
    def api_name(self):
        return self._api_name

    @property
    def api_version(self):
        return self._version

    @property
    def active(self):
        return self._active

    @abc.abstractmethod
    def Connect(self):
        pass

    @abc.abstractproperty
    def operator(self):
        pass

    @abc.abstractproperty
    def ns_label(self):
        pass

    @abc.abstractproperty
    def supports_one_platform(self):
        pass

    @abc.abstractproperty
    def location_label(self):
        pass

    def HttpClient(self):
        """The HTTP client to use to connect.

    May only be called inside the context represented by this ConnectionInfo

    Returns: An HTTP client specialized to connect in this context, or None if
    a standard HTTP client is appropriate.
    """
        return None

    def __enter__(self):
        self._active = True
        self._cm = self.Connect()
        return self._cm.__enter__()

    def __exit__(self, typ, value, traceback):
        self._active = False
        return self._cm.__exit__(typ, value, traceback)