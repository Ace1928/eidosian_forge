from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import subprocess
import sys
from googlecloudsdk.command_lib.code import run_subprocess
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core.console import console_io
from googlecloudsdk.core.util import platforms
from googlecloudsdk.core.util import times
import six
class KubeNamespace(object):
    """Context to create and tear down kubernetes namespace."""

    def __init__(self, namespace, context_name=None):
        """Initialize KubeNamespace.

    Args:
      namespace: (str) Namespace name.
      context_name: (str) Kubernetes context name.
    """
        self._namespace = namespace
        self._context_name = context_name
        self._delete_namespace = False

    def __enter__(self):
        if not _NamespaceExists(self._namespace, self._context_name):
            _CreateNamespace(self._namespace, self._context_name)
            self._delete_namespace = True

    def __exit__(self, exc_type, exc_value, tb):
        if self._delete_namespace:
            _DeleteNamespace(self._namespace, self._context_name)