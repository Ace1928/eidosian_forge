from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddBuildDockerArgs(argument_group, require_docker_image=False, update=False):
    """Adds additional argparse flags to a group for build docker options.

  Args:
    argument_group: argparse argument group to which build docker flag will
      be added.
    require_docker_image: If true, --dockerfile-image must be provided.
    update: Whether the command is update.
  """
    docker = argument_group.add_argument_group(help='Dockerfile build configuration flags')
    docker.add_argument('--dockerfile', required=True, help='Path of Dockerfile to use for builds in the repository.\n\nIf specified, a build config will be generated to run docker\nbuild using the specified file.\n\nThe filename is relative to the Dockerfile directory.\n')
    default_dir = '/'
    if update:
        default_dir = None
    docker.add_argument('--dockerfile-dir', default=default_dir, help='Location of the directory containing the Dockerfile in the repository.\n\nThe directory will also be used as the Docker build context.\n')
    docker_image_help_text = 'Docker image name to build.\n\nIf not specified, gcr.io/PROJECT/github.com/REPO_OWNER/REPO_NAME:$COMMIT_SHA will be used.\n\nUse a build configuration (cloudbuild.yaml) file for building multiple images in a single trigger.\n'
    if require_docker_image:
        docker_image_help_text = 'Docker image name to build.'
    docker.add_argument('--dockerfile-image', required=require_docker_image, help=docker_image_help_text)