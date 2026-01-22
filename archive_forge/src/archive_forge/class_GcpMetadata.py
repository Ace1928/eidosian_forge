from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GcpMetadata(_messages.Message):
    """GCP metadata associated with the resource, only applicable if the
  finding's cloud provider is Google Cloud Platform.

  Fields:
    folders: Output only. Contains a Folder message for each folder in the
      assets ancestry. The first folder is the deepest nested folder, and the
      last folder is the folder directly under the Organization.
    organization: The name of the organization that the resource belongs to.
    parent: The full resource name of resource's parent.
    parentDisplayName: The human readable name of resource's parent.
    project: The full resource name of project that the resource belongs to.
    projectDisplayName: The project ID that the resource belongs to.
  """
    folders = _messages.MessageField('GoogleCloudSecuritycenterV2Folder', 1, repeated=True)
    organization = _messages.StringField(2)
    parent = _messages.StringField(3)
    parentDisplayName = _messages.StringField(4)
    project = _messages.StringField(5)
    projectDisplayName = _messages.StringField(6)