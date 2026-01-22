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
class _SectionCode(_Section):
    """Contains the properties for the 'code' section."""

    def __init__(self):
        super(_SectionCode, self).__init__('code', hidden=True)
        self.minikube_event_timeout = self._Add('minikube_event_timeout', default='90s', hidden=True, help_text='Terminate the cluster start process if this amount of time has passed since the last minikube event.')
        self.minikube_path_override = self._Add('minikube_path_override', hidden=True, help_text='Location of minikube binary.')
        self.skaffold_path_override = self._Add('skaffold_path_override', hidden=True, help_text='Location of skaffold binary.')