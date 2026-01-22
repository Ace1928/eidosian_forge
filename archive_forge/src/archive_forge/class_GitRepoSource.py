from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitRepoSource(_messages.Message):
    """GitRepoSource describes a repo and ref of a code repository.

  Enums:
    RepoTypeValueValuesEnum: See RepoType below.

  Fields:
    bitbucketServerConfig: The full resource name of the bitbucket server
      config. Format:
      `projects/{project}/locations/{location}/bitbucketServerConfigs/{id}`.
    githubEnterpriseConfig: The full resource name of the github enterprise
      config. Format:
      `projects/{project}/locations/{location}/githubEnterpriseConfigs/{id}`.
      `projects/{project}/githubEnterpriseConfigs/{id}`.
    ref: The branch or tag to use. Must start with "refs/" (required).
    repoType: See RepoType below.
    repository: The connected repository resource name, in the format
      `projects/*/locations/*/connections/*/repositories/*`. Either `uri` or
      `repository` can be specified and is required.
    uri: The URI of the repo (e.g. https://github.com/user/repo.git). Either
      `uri` or `repository` can be specified and is required.
  """

    class RepoTypeValueValuesEnum(_messages.Enum):
        """See RepoType below.

    Values:
      UNKNOWN: The default, unknown repo type. Don't use it, instead use one
        of the other repo types.
      CLOUD_SOURCE_REPOSITORIES: A Google Cloud Source Repositories-hosted
        repo.
      GITHUB: A GitHub-hosted repo not necessarily on "github.com" (i.e.
        GitHub Enterprise).
      BITBUCKET_SERVER: A Bitbucket Server-hosted repo.
      GITLAB: A GitLab-hosted repo.
      BITBUCKET_CLOUD: A Bitbucket Cloud-hosted repo.
    """
        UNKNOWN = 0
        CLOUD_SOURCE_REPOSITORIES = 1
        GITHUB = 2
        BITBUCKET_SERVER = 3
        GITLAB = 4
        BITBUCKET_CLOUD = 5
    bitbucketServerConfig = _messages.StringField(1)
    githubEnterpriseConfig = _messages.StringField(2)
    ref = _messages.StringField(3)
    repoType = _messages.EnumField('RepoTypeValueValuesEnum', 4)
    repository = _messages.StringField(5)
    uri = _messages.StringField(6)