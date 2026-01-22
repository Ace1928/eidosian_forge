from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class RepositoryEventConfig(_messages.Message):
    """The configuration of a trigger that creates a build whenever an event
  from Repo API is received.

  Enums:
    RepositoryTypeValueValuesEnum: Output only. The type of the SCM vendor the
      repository points to.

  Fields:
    pullRequest: Filter to match changes in pull requests.
    push: Filter to match changes in refs like branches, tags.
    repository: The resource name of the Repo API resource.
    repositoryType: Output only. The type of the SCM vendor the repository
      points to.
  """

    class RepositoryTypeValueValuesEnum(_messages.Enum):
        """Output only. The type of the SCM vendor the repository points to.

    Values:
      REPOSITORY_TYPE_UNSPECIFIED: If unspecified, RepositoryType defaults to
        GITHUB.
      GITHUB: The SCM repo is GITHUB.
      GITHUB_ENTERPRISE: The SCM repo is GITHUB Enterprise.
      GITLAB_ENTERPRISE: The SCM repo is GITLAB Enterprise.
      BITBUCKET_DATA_CENTER: The SCM repo is BITBUCKET Data Center.
      BITBUCKET_CLOUD: The SCM repo is BITBUCKET Cloud.
    """
        REPOSITORY_TYPE_UNSPECIFIED = 0
        GITHUB = 1
        GITHUB_ENTERPRISE = 2
        GITLAB_ENTERPRISE = 3
        BITBUCKET_DATA_CENTER = 4
        BITBUCKET_CLOUD = 5
    pullRequest = _messages.MessageField('PullRequestFilter', 1)
    push = _messages.MessageField('PushFilter', 2)
    repository = _messages.StringField(3)
    repositoryType = _messages.EnumField('RepositoryTypeValueValuesEnum', 4)