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
class _ImportUrl(_BaseImport):
    """Class to perform operations on a URL import."""

    def __init__(self, full_path, name=None):
        full_path = self._ValidateUrl(full_path)
        name = name if name else full_path
        super(_ImportUrl, self).__init__(full_path, name)

    def GetBaseName(self):
        if self.base_name is None:
            self.base_name = posixpath.basename(six.moves.urllib.parse.urlparse(self.full_path).path)
        return self.base_name

    def Exists(self):
        if self.content:
            return True
        return self._RetrieveContent(raise_exceptions=False)

    def GetContent(self):
        if self.content is None:
            self._RetrieveContent()
        return self.content

    def _RetrieveContent(self, raise_exceptions=True):
        """Helper function for both Exists and GetContent.

    Args:
      raise_exceptions: Set to false if you just want to know if the file
          actually exists.

    Returns:
      True if we successfully got the content of the URL. Returns False is
      raise_exceptions is False.

    Raises:
      HTTPError: If raise_exceptions is True, will raise exceptions for 4xx or
          5xx response codes instead of returning False.
    """
        r = requests.get(self.full_path)
        try:
            r.raise_for_status()
        except requests.exceptions.HTTPError as e:
            if raise_exceptions:
                raise e
            return False
        self.content = r.text
        return True

    def BuildChildPath(self, child_path):
        if _IsUrl(child_path):
            return child_path
        return six.moves.urllib.parse.urljoin(self.full_path, child_path)

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