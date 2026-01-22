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
class ExternalClusterContext(object):
    """Do nothing context manager for external clusters."""

    def __init__(self, kube_context):
        self._kube_context = kube_context

    def __enter__(self):
        return ExternalCluster(self._kube_context)

    def __exit__(self, exc_type, exc_value, tb):
        pass