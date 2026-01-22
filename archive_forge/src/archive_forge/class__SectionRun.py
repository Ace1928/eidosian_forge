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
class _SectionRun(_Section):
    """Contains the properties for the 'run' section."""

    def __init__(self):
        super(_SectionRun, self).__init__('run')
        self.region = self._Add('region', help_text='Default region to use when working with Cloud Run resources. When a `--region` flag is required but not provided, the command will fall back to this value, if set.')
        self.namespace = self._Add('namespace', help_text='Specific to working with Cloud on GKE or a Kubernetes cluster: Kubernetes namespace for the resource.', hidden=True)
        self.cluster = self._Add('cluster', help_text='ID of the cluster or fully qualified identifier for the cluster')
        self.cluster_location = self._Add('cluster_location', help_text='Zone or region in which the cluster is located.')
        self.platform = self._Add('platform', choices=['gke', 'managed', 'kubernetes'], default='managed', help_text='Target platform for running commands.')