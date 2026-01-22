from __future__ import print_function
from __future__ import unicode_literals
import collections
import contextlib
import gzip
import json
import keyword
import logging
import os
import re
import tempfile
import six
from six.moves import urllib_parse
import six.moves.urllib.error as urllib_error
import six.moves.urllib.request as urllib_request
def _ComputePaths(package, version, root_url, service_path):
    """Compute the base url and base path.

    Attributes:
      package: name field of the discovery, i.e. 'storage' for storage service.
      version: version of the service, i.e. 'v1'.
      root_url: root url of the service, i.e. 'https://www.googleapis.com/'.
      service_path: path of the service under the rool url, i.e. 'storage/v1/'.

    Returns:
      base url: string, base url of the service,
        'https://www.googleapis.com/storage/v1/' for the storage service.
      base path: string, common prefix of service endpoints after the base url.
    """
    full_path = urllib_parse.urljoin(root_url, service_path)
    api_path_component = '/'.join((package, version, ''))
    if api_path_component not in full_path:
        return (full_path, '')
    prefix, _, suffix = full_path.rpartition(api_path_component)
    return (prefix + api_path_component, suffix)