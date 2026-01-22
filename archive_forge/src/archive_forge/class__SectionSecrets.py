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
class _SectionSecrets(_Section):
    """Contains the properties for the 'secrets' section."""

    def __init__(self):
        super(_SectionSecrets, self).__init__('secrets')
        self.replication_policy = self._Add('replication-policy', choices=['automatic', 'user-managed'], help_text='The type of replication policy to apply to secrets. Allowed values are "automatic" and "user-managed". If user-managed then locations must also be provided.')
        self.locations = self._Add('locations', help_text='A comma separated list of the locations to replicate secrets to. Only applies to secrets with a user-managed policy.')