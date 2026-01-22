from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
import collections
import contextlib
import io
import ipaddress
import os
import re
from googlecloudsdk.api_lib.composer import util as api_util
from googlecloudsdk.api_lib.container import api_adapter as gke_api_adapter
from googlecloudsdk.api_lib.container import util as gke_util
from googlecloudsdk.api_lib.storage import storage_api
from googlecloudsdk.api_lib.storage import storage_util
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.composer import parsers
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions as core_exceptions
from googlecloudsdk.core import execution_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.resource import resource_printer
from googlecloudsdk.core.util import encoding
from googlecloudsdk.core.util import files
import six
def FetchKubectlNamespace(env_image_version):
    """Checks environment for valid namespace options.

  First checks for the existence of a kubectl namespace based on the env image
  version. If namespace does not exist, then return the 'default' namespace.

  Args:
    env_image_version: str, the environment image version string.

  Returns:
    The namespace string to apply to any `environments run` commands.
  """
    image_version_ns_prefix = ConvertImageVersionToNamespacePrefix(env_image_version)
    args = ['get', 'namespace', '--all-namespaces', '--sort-by=.metadata.creationTimestamp', '--output', 'jsonpath={range .items[*]}{.metadata.name}{"\\t"}{.status.phase}{"\\n"}', '--ignore-not-found=true']
    ns_output = io.StringIO()
    RunKubectlCommand(args, ns_output.write, log.err.write)
    namespaces = reversed(ns_output.getvalue().split('\n'))
    for ns_entry in namespaces:
        ns_parts = ns_entry.split('\t') if ns_entry.strip() else None
        if ns_parts and ns_parts[1].lower() == NAMESPACE_STATUS_ACTIVE and ns_parts[0].startswith(image_version_ns_prefix):
            return ns_parts[0]
    return DEFAULT_NAMESPACE