from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import glob
import os
import posixpath
import re
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.deployment_manager import exceptions
from googlecloudsdk.api_lib.util import exceptions as api_exceptions
from googlecloudsdk.core import yaml
import googlecloudsdk.core.properties
from googlecloudsdk.core.util import files
import requests
import six
import six.moves.urllib.parse
@staticmethod
def _ValidateUrl(url):
    """Make sure the url fits the format we expect."""
    parsed_url = six.moves.urllib.parse.urlparse(url)
    if parsed_url.scheme not in ('http', 'https'):
        raise exceptions.ConfigError("URL '%s' scheme was '%s'; it must be either 'https' or 'http'." % (url, parsed_url.scheme))
    if not parsed_url.path or parsed_url.path == '/':
        raise exceptions.ConfigError("URL '%s' doesn't have a path." % url)
    if parsed_url.params or parsed_url.query or parsed_url.fragment:
        raise exceptions.ConfigError("URL '%s' should only have a path, no params, queries, or fragments." % url)
    return url