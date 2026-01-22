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
class _SectionArtifacts(_Section):
    """Contains the properties for the 'artifacts' section."""

    def __init__(self):
        super(_SectionArtifacts, self).__init__('artifacts')
        self.repository = self._Add('repository', help_text='Default repository to use when working with Artifact Registry resources. When a `repository` value is required but not provided, the command will fall back to this value, if set.')
        self.location = self._Add('location', help_text='Default location to use when working with Artifact Registry resources. When a `location` value is required but not provided, the command will fall back to this value, if set. If this value is unset, the default location is `global` when `location` value is optional.')
        self.registry_endpoint_prefix = self._Add('registry_endpoint_prefix', default='', hidden=True, help_text='Default prefix to use while interacting with Artifact Registry resources.')
        self.domain = self._Add('domain', default='pkg.dev', hidden=True, help_text='Default domain endpoint to use while interacting with Artifact Registry Docker resources.')
        self.gcr_host = self._Add('gcr_host', default='gcr.io', hidden=True, help_text='Default host to use while interacting with Container Registry Docker resources.')