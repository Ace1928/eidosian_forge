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
def _StartMinikubeCluster(cluster_name, vm_driver, debug=False):
    """Starts a minikube cluster."""
    try:
        if not _IsMinikubeClusterUp(cluster_name):
            cmd = [_FindMinikube(), 'start', '-p', cluster_name, '--keep-context', '--interactive=false', '--delete-on-failure', '--install-addons=false', '--output=json']
            if vm_driver:
                cmd.append('--vm-driver=' + vm_driver)
                if vm_driver == 'docker':
                    cmd.append('--container-runtime=docker')
            if debug:
                cmd.extend(['--alsologtostderr', '-v8'])
            start_msg = "Starting development environment '%s' ..." % cluster_name
            event_timeout = times.ParseDuration(properties.VALUES.code.minikube_event_timeout.Get(required=True)).total_seconds
            with console_io.ProgressBar(start_msg) as progress_bar:
                for json_obj in run_subprocess.StreamOutputJson(cmd, event_timeout_sec=event_timeout, show_stderr=debug):
                    if debug:
                        print('minikube', json_obj)
                    _HandleMinikubeStatusEvent(progress_bar, json_obj)
    except Exception as e:
        six.reraise(MinikubeStartError, e, sys.exc_info()[2])