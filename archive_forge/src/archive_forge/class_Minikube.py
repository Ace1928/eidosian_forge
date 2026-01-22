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
class Minikube(object):
    """Starts and stops a minikube cluster."""

    def __init__(self, cluster_name, stop_cluster=True, vm_driver=None, debug=False):
        self._cluster_name = cluster_name
        self._stop_cluster = stop_cluster
        self._vm_driver = vm_driver
        self._debug = debug

    def __enter__(self):
        _StartMinikubeCluster(self._cluster_name, self._vm_driver, self._debug)
        return MinikubeCluster(self._cluster_name, self._vm_driver == 'docker')

    def __exit__(self, exc_type, exc_value, tb):
        if self._stop_cluster:
            _StopMinikube(self._cluster_name, self._debug)