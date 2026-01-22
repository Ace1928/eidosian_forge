from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitLabEventsConfig(_messages.Message):
    """GitLabEventsConfig describes the configuration of a trigger that creates
  a build whenever a GitLab event is received.

  Fields:
    gitlabConfig: Output only. The GitLabConfig specified in the
      gitlab_config_resource field.
    gitlabConfigResource: The GitLab config resource that this trigger config
      maps to.
    projectNamespace: Namespace of the GitLab project.
    pullRequest: Filter to match changes in pull requests.
    push: Filter to match changes in refs like branches, tags.
  """
    gitlabConfig = _messages.MessageField('GitLabConfig', 1)
    gitlabConfigResource = _messages.StringField(2)
    projectNamespace = _messages.StringField(3)
    pullRequest = _messages.MessageField('PullRequestFilter', 4)
    push = _messages.MessageField('PushFilter', 5)