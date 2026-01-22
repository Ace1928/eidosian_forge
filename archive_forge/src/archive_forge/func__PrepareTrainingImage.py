from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.ai.custom_jobs import local_util
from googlecloudsdk.command_lib.ai.docker import build as docker_build
from googlecloudsdk.command_lib.ai.docker import utils as docker_utils
from googlecloudsdk.command_lib.util.apis import arg_utils
from googlecloudsdk.core import log
from googlecloudsdk.core.util import files
def _PrepareTrainingImage(project, job_name, base_image, local_package, script, output_image_name, python_module=None, **kwargs):
    """Build a training image from local package and push it to Cloud for later usage."""
    output_image = output_image_name or docker_utils.GenerateImageName(base_name=job_name, project=project, is_gcr=True)
    docker_build.BuildImage(base_image=base_image, host_workdir=files.ExpandHomeDir(local_package), main_script=script, python_module=python_module, output_image_name=output_image, **kwargs)
    log.status.Print('\nA custom container image is built locally.\n')
    push_command = ['docker', 'push', output_image]
    docker_utils.ExecuteDockerCommand(push_command)
    log.status.Print('\nCustom container image [{}] is created for your custom job.\n'.format(output_image))
    return output_image