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
class _SectionTest(_Section):
    """Contains the properties for the 'test' section."""

    def __init__(self):
        super(_SectionTest, self).__init__('test')
        self.results_base_url = self._Add('results_base_url', hidden=True)
        self.matrix_status_interval = self._Add('matrix_status_interval', hidden=True)
        self.feature_flag = self._Add('feature_flag', hidden=True, internal=True, is_feature_flag=True, help_text='Run `gcloud meta test --feature-flag` to test the value of this feature flag.')