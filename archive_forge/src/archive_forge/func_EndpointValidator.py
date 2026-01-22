from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import enum
import functools
import os
import re
import sys
import textwrap
from googlecloudsdk.core import argv_utils
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core.configurations import named_configs
from googlecloudsdk.core.configurations import properties_file as prop_files_lib
from googlecloudsdk.core.docker import constants as const_lib
from googlecloudsdk.core.resource import resource_printer_types as formats
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import http_proxy_types
from googlecloudsdk.core.util import scaled_integer
from googlecloudsdk.generated_clients.apis import apis_map
import six
def EndpointValidator(self, value):
    """Checks to see if the endpoint override string is valid."""
    if value is None:
        return
    if not _VALID_ENDPOINT_OVERRIDE_REGEX.match(value):
        raise InvalidValueError("The endpoint_overrides property must be an absolute URI beginning with http:// or https:// and ending with a trailing '/'. [{value}] is not a valid endpoint override.".format(value=value))