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
def AddKubectlNamespace(namespace, kubectl_args):
    """Adds namespace arguments to the provided list of kubectl args.

  If a namespace arg is not already present, insert `--namespace <namespace>`
  after the `kubectl` command and before all other arg elements.

  Resulting in this general format:
    ['kubectl', '--namespace', 'namespace_foo', ... <remaining args> ... ]

  Args:
    namespace: name of the namespace scope
    kubectl_args: list of kubectl command arguments. Expects that the first
      element will be the `kubectl` command, followed by all additional
      arguments.

  Returns:
    list of kubectl args with the additional namespace args (if necessary).
  """
    if namespace is None:
        return kubectl_args
    if {NAMESPACE_ARG_NAME, NAMESPACE_ARG_ALIAS}.isdisjoint(set(kubectl_args)):
        idx = 0
        if kubectl_args and _KUBECTL_COMPONENT_NAME in kubectl_args[0]:
            idx = 1
        for new_arg in [namespace, NAMESPACE_ARG_NAME]:
            kubectl_args.insert(idx, new_arg)
    return kubectl_args