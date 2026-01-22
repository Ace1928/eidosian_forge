from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudbuildProjectsLocationsBitbucketServerConfigsPatchRequest(_messages.Message):
    """A CloudbuildProjectsLocationsBitbucketServerConfigsPatchRequest object.

  Fields:
    bitbucketServerConfig: A BitbucketServerConfig resource to be passed as
      the request body.
    name: The resource name for the config.
    updateMask: Update mask for the resource. If this is set, the server will
      only update the fields specified in the field mask. Otherwise, a full
      update of the mutable resource fields will be performed.
  """
    bitbucketServerConfig = _messages.MessageField('BitbucketServerConfig', 1)
    name = _messages.StringField(2, required=True)
    updateMask = _messages.StringField(3)