from __future__ import absolute_import
from __future__ import division
from __future__ import unicode_literals
from googlecloudsdk.api_lib.cloudbuild import cloudbuild_util
from googlecloudsdk.calliope import arg_parsers
from googlecloudsdk.calliope import exceptions as c_exceptions
from googlecloudsdk.command_lib.builds import flags as build_flags
def ParseGitRepoSource(trigger, args, messages, required=False):
    """Parses git repo source flags.

  Args:
    trigger: The trigger to populate.
    args: An argparse arguments object.
    messages: A Cloud Build messages module.
    required: Whether or not the repository info is required.
  """
    if required and (not args.repo) and (not args.repository):
        raise c_exceptions.RequiredArgumentException('REPO', '--repo or --repository is required when specifying a --dockerfile or --build-config.')
    if not args.repo and (not args.repository):
        if args.branch or args.tag:
            raise c_exceptions.RequiredArgumentException('REPO', '--repo or --repository is required when specifying a --branch or --tag.')
        return
    if not args.branch and (not args.tag):
        raise c_exceptions.RequiredArgumentException('REVISION', '--branch or --tag is required when specifying a --repository or --repo.')
    if args.branch:
        ref = 'refs/heads/' + args.branch
    else:
        ref = 'refs/tags/' + args.tag
    parsed_git_repo_source_repo_type = None if not args.repo_type else messages.GitRepoSource.RepoTypeValueValuesEnum(args.repo_type)
    trigger.sourceToBuild = messages.GitRepoSource(repository=args.repository, uri=args.repo, ref=ref, repoType=parsed_git_repo_source_repo_type, githubEnterpriseConfig=args.github_enterprise_config, bitbucketServerConfig=args.bitbucket_server_config)
    parsed_git_file_source_repo_type = None if not args.repo_type else messages.GitFileSource.RepoTypeValueValuesEnum(args.repo_type)
    if args.build_config:
        trigger.gitFileSource = messages.GitFileSource(repository=args.repository, path=args.build_config, uri=args.repo, revision=ref, repoType=parsed_git_file_source_repo_type, githubEnterpriseConfig=args.github_enterprise_config, bitbucketServerConfig=args.bitbucket_server_config)