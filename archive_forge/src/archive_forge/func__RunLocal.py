from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import contextlib
import subprocess
from googlecloudsdk.calliope import base
from googlecloudsdk.command_lib.code import cross_platform_temp_file
from googlecloudsdk.command_lib.code import flags
from googlecloudsdk.command_lib.code import kubernetes
from googlecloudsdk.command_lib.code import local
from googlecloudsdk.command_lib.code import local_files
from googlecloudsdk.command_lib.code import run_subprocess
from googlecloudsdk.command_lib.code import skaffold
from googlecloudsdk.command_lib.code import yaml_helper
from googlecloudsdk.command_lib.code.cloud import artifact_registry
from googlecloudsdk.command_lib.code.cloud import cloud
from googlecloudsdk.command_lib.code.cloud import cloud_files
from googlecloudsdk.command_lib.code.cloud import cloudrun
from googlecloudsdk.core import config
from googlecloudsdk.core import exceptions
from googlecloudsdk.core import properties
from googlecloudsdk.core import yaml
from googlecloudsdk.core.updater import update_manager
from googlecloudsdk.core.util import files as file_utils
import portpicker
import six
def _RunLocal(self, args):
    settings = local.AssembleSettings(args, self.ReleaseTrack())
    local_file_generator = local_files.LocalRuntimeFiles(settings)
    kubernetes_config = six.ensure_text(local_file_generator.KubernetesConfig())
    namespace = getattr(args, 'namespace', None)
    _EnsureDockerRunning()
    with _DeployTempFile(kubernetes_config) as kubernetes_file:
        skaffold_config = six.ensure_text(local_file_generator.SkaffoldConfig(kubernetes_file.name))
        skaffold_event_port = args.skaffold_events_port or portpicker.pick_unused_port()
        with _SkaffoldTempFile(skaffold_config) as skaffold_file, self._GetKubernetesEngine(args) as kube_context, self._WithKubeNamespace(namespace, kube_context.context_name), _SetImagePush(skaffold_file, kube_context.shared_docker) as patched_skaffold_file, self._SkaffoldProcess(patched_skaffold_file, kube_context, namespace, skaffold_event_port) as running_process, skaffold.PrintUrlThreadContext(settings.service_name, skaffold_event_port):
            running_process.wait()