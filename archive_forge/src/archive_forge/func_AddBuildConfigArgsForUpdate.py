from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddBuildConfigArgsForUpdate(flag_config, has_build_config=False, has_file_source=False, require_docker_image=False):
    """Adds additional argparse flags to a group for build configuration options for update command.

  Args:
    flag_config: Argparse argument group. Additional flags will be added to this
      group to cover common build configuration settings.
    has_build_config: Whether it is possible for the trigger to have
      filename.
    has_file_source: Whether it is possible for the trigger to have
      git_file_source.
    require_docker_image: If true, dockerfile image must be provided.

  Returns:
    build_config: A build config.
  """
    substitutions = flag_config.add_mutually_exclusive_group()
    AddSubstitutionUpdatingFlags(substitutions)
    build_config = flag_config.add_mutually_exclusive_group()
    if has_build_config:
        build_config.add_argument('--build-config', metavar='PATH', help='  Path to a YAML or JSON file containing the build configuration in the repository.\n\n  For more details, see: https://cloud.google.com/cloud-build/docs/build-config\n  ')
    build_config.add_argument('--inline-config', metavar='PATH', help='      Local path to a YAML or JSON file containing a build configuration.\n    ')
    if has_file_source:
        AddGitFileSourceArgs(build_config)
    AddBuildDockerArgs(build_config, require_docker_image=require_docker_image, update=True)
    return build_config