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
@contextlib.contextmanager
def _OverrideEndpointOverrides(api_name, override):
    """Context manager to override an API's endpoint overrides for a while.

  Args:
    api_name: str, Name of the API to modify.
    override: str, New value for the endpoint.

  Yields:
    None.
  """
    endpoint_property = getattr(properties.VALUES.api_endpoint_overrides, api_name)
    old_endpoint = endpoint_property.Get()
    try:
        endpoint_property.Set(override)
        yield
    finally:
        endpoint_property.Set(old_endpoint)