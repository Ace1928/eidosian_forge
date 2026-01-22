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
class _SectionFunctions(_Section):
    """Contains the properties for the 'functions' section."""

    def __init__(self):
        super(_SectionFunctions, self).__init__('functions')
        self.region = self._Add('region', default='us-central1', help_text='Default region to use when working with Cloud Functions resources. When a `--region` flag is required but not provided, the command will fall back to this value, if set. To see valid choices, run `gcloud beta functions regions list`.', completer='googlecloudsdk.command_lib.functions.flags:LocationsCompleter')
        self.gen2 = self._AddBool('gen2', default=False, help_text='Default environment to use when working with Cloud Functions resources. When neither `--gen2` nor `--no-gen2` is provided, the decision of whether to use Generation 2 falls back to this value if set.')
        self.v2 = self._AddBool('v2', default=False, hidden=True, help_text='DEPRECATED. Use `functions/gen2` instead. This property will be removed in a future release.')