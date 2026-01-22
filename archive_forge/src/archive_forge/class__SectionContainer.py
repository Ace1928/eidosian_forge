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
class _SectionContainer(_Section):
    """Contains the properties for the 'container' section."""

    def __init__(self):
        super(_SectionContainer, self).__init__('container')
        self.cluster = self._Add('cluster', help_text='Name of the cluster to use by default when working with Kubernetes Engine.')
        self.use_client_certificate = self._AddBool('use_client_certificate', default=False, help_text="If True, use the cluster's client certificate to authenticate to the cluster API server.")
        self.use_app_default_credentials = self._AddBool('use_application_default_credentials', default=False, help_text='If True, use application default credentials to authenticate to the cluster API server.')
        self.build_timeout = self._Add('build_timeout', validator=_BuildTimeoutValidator, help_text='Timeout, in seconds, to wait for container builds to complete.')
        self.build_check_tag = self._AddBool('build_check_tag', default=True, hidden=True, help_text='If True, validate that the --tag value to container builds submit is in the gcr.io or *.gcr.io namespace.')