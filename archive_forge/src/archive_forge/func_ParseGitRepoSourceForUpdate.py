from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def ParseGitRepoSourceForUpdate(trigger, args, messages, update_mask):
    """Parses git repo source flags for update command.

  Args:
    trigger: The trigger to populate.
    args: An argparse arguments object.
    messages: A Cloud Build messages module.
    update_mask: The fields to be updated.
  """
    trigger.sourceToBuild = messages.GitRepoSource()
    reporef = None
    if args.source_to_build_branch:
        reporef = 'refs/heads/' + args.source_to_build_branch
    elif args.source_to_build_tag:
        reporef = 'refs/tags/' + args.source_to_build_tag
    if reporef:
        trigger.sourceToBuild.ref = reporef
    if args.source_to_build_repository:
        trigger.sourceToBuild.repository = args.source_to_build_repository
        update_mask.append('source_to_build.uri')
        update_mask.append('source_to_build.repo_type')
        update_mask.append('source_to_build.github_enterprise_config')
    elif args.source_to_build_uri or args.source_to_build_github_enterprise_config or args.source_to_build_repo_type:
        trigger.sourceToBuild.uri = args.source_to_build_uri
        trigger.sourceToBuild.githubEnterpriseConfig = args.source_to_build_github_enterprise_config
        if args.source_to_build_repo_type:
            trigger.sourceToBuild.repoType = messages.GitRepoSource.RepoTypeValueValuesEnum(args.source_to_build_repo_type)
        update_mask.append('source_to_build.repository')