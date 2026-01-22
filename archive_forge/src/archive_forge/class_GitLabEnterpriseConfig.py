from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GitLabEnterpriseConfig(_messages.Message):
    """GitLabEnterpriseConfig represents the configuration for a
  GitLabEnterprise integration.

  Fields:
    hostUri: Immutable. The URI of the GitlabEnterprise host.
    serviceDirectoryConfig: The Service Directory configuration to be used
      when reaching out to the GitLab Enterprise instance.
    sslCa: The SSL certificate to use in requests to GitLab Enterprise
      instances.
  """
    hostUri = _messages.StringField(1)
    serviceDirectoryConfig = _messages.MessageField('ServiceDirectoryConfig', 2)
    sslCa = _messages.StringField(3)