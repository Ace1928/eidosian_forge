from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitHubEnterpriseApp(_messages.Message):
    """RPC response object returned by the GetGitHubEnterpriseApp RPC method.

  Fields:
    name: Name of the GitHub App
    slug: Slug (URL friendly name) of the GitHub App. This can be found on the
      settings page for the GitHub App (e.g.
      https://github.com/settings/apps/:app_slug) GitHub docs:
      https://docs.github.com/en/free-pro-team@latest/rest/reference/apps#get-
      an-app
  """
    name = _messages.StringField(1)
    slug = _messages.StringField(2)