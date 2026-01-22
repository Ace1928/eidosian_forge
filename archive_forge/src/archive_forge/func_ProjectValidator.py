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
def ProjectValidator(project):
    """Checks to see if the project string is valid."""
    if project is None:
        return
    if not isinstance(project, six.string_types):
        raise InvalidValueError('project must be a string')
    if project == '':
        raise InvalidProjectError('The project property is set to the empty string, which is invalid.')
    if _VALID_PROJECT_REGEX.match(project):
        return
    if _LooksLikeAProjectName(project):
        raise InvalidProjectError('The project property must be set to a valid project ID, not the project name [{value}]'.format(value=project))
    raise InvalidProjectError('The project property must be set to a valid project ID, [{value}] is not a valid project ID.'.format(value=project))