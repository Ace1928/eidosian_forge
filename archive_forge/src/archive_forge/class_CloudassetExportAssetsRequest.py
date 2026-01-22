from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudassetExportAssetsRequest(_messages.Message):
    """A CloudassetExportAssetsRequest object.

  Fields:
    exportAssetsRequest: A ExportAssetsRequest resource to be passed as the
      request body.
    parent: Required. The relative name of the root asset. This can only be an
      organization number (such as "organizations/123"), a project ID (such as
      "projects/my-project-id"), or a project number (such as
      "projects/12345"), or a folder number (such as "folders/123").
  """
    exportAssetsRequest = _messages.MessageField('ExportAssetsRequest', 1)
    parent = _messages.StringField(2, required=True)