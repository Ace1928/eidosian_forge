from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.command_lib.ai.docker import utils
from googlecloudsdk.core import config
def RunContainer(image_name, enable_gpu=False, service_account_key=None, run_args=None, user_args=None):
    """Calls `docker run` on a given image with specified arguments.

  Args:
    image_name: (str) Name or ID of Docker image to run.
    enable_gpu: (bool) Whether to use GPU
    service_account_key: (str) Json file of a service account key  auth.
    run_args: (List[str]) Extra custom options to apply to `docker run` after
      our defaults.
    user_args: (List[str]) Extra user defined arguments to supply to the
      entrypoint.
  """
    if run_args is None:
        run_args = []
    if user_args is None:
        user_args = []
    run_opts = _DockerRunOptions(enable_gpu=enable_gpu, service_account_key=service_account_key, extra_run_opts=run_args)
    command = ['docker', 'run'] + run_opts + [image_name] + user_args
    utils.ExecuteDockerCommand(command)