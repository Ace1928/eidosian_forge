from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def AddGitFileSourceArgs(argument_group):
    """Adds additional argparse flags to a group for git file source options.

  Args:
    argument_group: Argparse argument group to which git file source flag will
      be added.
  """
    git_file_source = argument_group.add_argument_group(help='Build file source flags')
    repo_source = git_file_source.add_mutually_exclusive_group()
    repo_source.add_argument('--git-file-source-repository', hidden=True, help='Repository resource (2nd gen) to use, in the format "projects/*/locations/*/connections/*/repositories/*".\n')
    v1_gen_source = repo_source.add_argument_group()
    v1_gen_source.add_argument('--git-file-source-path', metavar='PATH', help='The file in the repository to clone when trigger is invoked.\n')
    v1_gen_repo_info = v1_gen_source.add_argument_group()
    v1_gen_repo_info.add_argument('--git-file-source-uri', required=True, metavar='URL', help='The URI of the repository to clone when trigger is invoked.\n')
    v1_gen_repo_info.add_argument('--git-file-source-repo-type', required=True, help='The type of the repository to clone when trigger is invoked.\n')
    v1_gen_source.add_argument('--git-file-source-github-enterprise-config', help='The resource name of the GitHub Enterprise config that should be applied to this source.\n')
    ref_config = git_file_source.add_mutually_exclusive_group()
    ref_config.add_argument('--git-file-source-branch', help='The branch of the repository to clone when trigger is invoked.\n')
    ref_config.add_argument('--git-file-source-tag', help='The tag of the repository to clone when trigger is invoked.\n')