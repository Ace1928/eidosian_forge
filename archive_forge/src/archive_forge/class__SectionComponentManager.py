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
class _SectionComponentManager(_Section):
    """Contains the properties for the 'component_manager' section."""

    def __init__(self):
        super(_SectionComponentManager, self).__init__('component_manager')
        self.additional_repositories = self._Add('additional_repositories', help_text='Comma separated list of additional repositories to check for components.  This property is automatically managed by the `gcloud components repositories` commands.')
        self.disable_update_check = self._AddBool('disable_update_check', help_text='If True, Google Cloud CLI will not automatically check for updates.')
        self.disable_warning = self._AddBool('disable_warning', hidden=True, help_text='If True, Google Cloud CLI will not display warning messages about overridden configurations.')
        self.fixed_sdk_version = self._Add('fixed_sdk_version', hidden=True)
        self.snapshot_url = self._Add('snapshot_url', hidden=True)
        self.original_snapshot_url = self._Add('original_snapshot_url', internal=True, hidden=True, help_text='Snapshot URL when this installation is firstly installed.', default='https://dl.google.com/dl/cloudsdk/channels/rapid/components-2.json')