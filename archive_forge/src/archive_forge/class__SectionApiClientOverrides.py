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
class _SectionApiClientOverrides(_Section):
    """Contains the properties for the 'api-client-overrides' section.

  This overrides the API client version to use when talking to this API.
  """

    def __init__(self):
        super(_SectionApiClientOverrides, self).__init__('api_client_overrides', hidden=True)
        self.alloydb = self._Add('alloydb')
        self.appengine = self._Add('appengine')
        self.baremetalsolution = self._Add('baremetalsolution')
        self.cloudidentity = self._Add('cloudidentity')
        self.compute = self._Add('compute')
        self.compute_alpha = self._Add('compute/alpha')
        self.compute_beta = self._Add('compute/beta')
        self.compute_v1 = self._Add('compute/v1')
        self.container = self._Add('container')
        self.speech = self._Add('speech')
        self.sql = self._Add('sql')
        self.storage = self._Add('storage')
        self.run = self._Add('run')
        self.scc = self._Add('securitycenter')
        self.cloudresourcemanager = self._Add('cloudresourcemanager')
        self.workstations = self._Add('workstations')