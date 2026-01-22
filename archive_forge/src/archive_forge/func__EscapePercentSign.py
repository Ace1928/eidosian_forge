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
def _EscapePercentSign(value):
    """Escape '%' in property value.

  Do nothing if value contains '%%', i.e. value was escaped by user.

  Args:
    value: property value

  Returns:
    str, value with escaped % sign
  """
    if not isinstance(value, six.string_types) or '%%' in value:
        return value
    elif '%' in value:
        return value.replace('%', '%%')
    else:
        return value