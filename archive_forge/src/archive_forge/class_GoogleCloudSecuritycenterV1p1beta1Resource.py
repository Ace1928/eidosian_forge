from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudSecuritycenterV1p1beta1Resource(_messages.Message):
    """Information related to the Google Cloud resource.

  Fields:
    folders: Output only. Contains a Folder message for each folder in the
      assets ancestry. The first folder is the deepest nested folder, and the
      last folder is the folder directly under the Organization.
    name: The full resource name of the resource. See:
      https://cloud.google.com/apis/design/resource_names#full_resource_name
    parent: The full resource name of resource's parent.
    parentDisplayName: The human readable name of resource's parent.
    project: The full resource name of project that the resource belongs to.
    projectDisplayName: The project id that the resource belongs to.
  """
    folders = _messages.MessageField('GoogleCloudSecuritycenterV1p1beta1Folder', 1, repeated=True)
    name = _messages.StringField(2)
    parent = _messages.StringField(3)
    parentDisplayName = _messages.StringField(4)
    project = _messages.StringField(5)
    projectDisplayName = _messages.StringField(6)