from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import io
import json
import logging
import string
from apitools.base.py import exceptions as apitools_exceptions
from googlecloudsdk.api_lib.util import resource as resource_util
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import log
from googlecloudsdk.core import properties
from googlecloudsdk.core.resource import resource_lex
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.resource import resource_property
from googlecloudsdk.core.util import encoding
import six
def _ExtractUrlResourceAndInstanceNames(self, http_error):
    """Extracts the url resource type and instance names from the HttpError."""
    self.url = http_error.url
    if not self.url:
        return
    try:
        name, version, resource_path = resource_util.SplitEndpointUrl(self.url)
    except resource_util.InvalidEndpointException:
        return
    if name:
        self.api_name = name
    if version:
        self.api_version = version
    resource_parts = resource_path.split('/')
    if not 1 < len(resource_parts) < 4:
        return
    self.resource_name = resource_parts[0]
    instance_name = resource_parts[1]
    self.instance_name = instance_name.split('?')[0]
    self.resource_item = '{} instance'.format(self.resource_name)