from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetQueryAssetsRequest(_messages.Message):
    """A CloudassetQueryAssetsRequest object.

  Fields:
    parent: Required. The relative name of the root asset. This can only be an
      organization number (such as "organizations/123"), a project ID (such as
      "projects/my-project-id"), or a project number (such as
      "projects/12345"), or a folder number (such as "folders/123"). Only
      assets belonging to the `parent` will be returned.
    queryAssetsRequest: A QueryAssetsRequest resource to be passed as the
      request body.
  """
    parent = _messages.StringField(1, required=True)
    queryAssetsRequest = _messages.MessageField('QueryAssetsRequest', 2)