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
@contextlib.contextmanager
def _SetImagePush(skaffold_file, shared_docker):
    """Set build.local.push value in skaffold file.

  Args:
    skaffold_file: Skaffold file handle.
    shared_docker: Boolean that is true if docker instance is shared between the
      kubernetes cluster and local docker builder.

  Yields:
    Path of skaffold file with build.local.push value set to the proper value.
  """
    if not shared_docker:
        yield skaffold_file
    else:
        skaffold_yaml = yaml.load_path(skaffold_file.name)
        local_block = yaml_helper.GetOrCreate(skaffold_yaml, ('build', 'local'))
        local_block['push'] = False
        with _SkaffoldTempFile(yaml.dump(skaffold_yaml)) as patched_skaffold_file:
            yield patched_skaffold_file