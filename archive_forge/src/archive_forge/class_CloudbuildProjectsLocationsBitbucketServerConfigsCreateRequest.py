from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsBitbucketServerConfigsCreateRequest(_messages.Message):
    """A CloudbuildProjectsLocationsBitbucketServerConfigsCreateRequest object.

  Fields:
    bitbucketServerConfig: A BitbucketServerConfig resource to be passed as
      the request body.
    bitbucketServerConfigId: Optional. The ID to use for the
      BitbucketServerConfig, which will become the final component of the
      BitbucketServerConfig's resource name. bitbucket_server_config_id must
      meet the following requirements: + They must contain only alphanumeric
      characters and dashes. + They can be 1-64 characters long. + They must
      begin and end with an alphanumeric character.
    parent: Required. Name of the parent resource.
  """
    bitbucketServerConfig = _messages.MessageField('BitbucketServerConfig', 1)
    bitbucketServerConfigId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)