from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedGroupConfig(_messages.Message):
    """Specifies the resources used to actively manage an instance group.

  Fields:
    instanceGroupManagerName: Output only. The name of the Instance Group
      Manager for this group.
    instanceGroupManagerUri: Output only. The partial URI to the instance
      group manager for this group. E.g. projects/my-project/regions/us-
      central1/instanceGroupManagers/my-igm.
    instanceTemplateName: Output only. The name of the Instance Template used
      for the Managed Instance Group.
    instanceTemplateUri: Optional. Output only. Partial URI of the Instance
      Template. Example:
      projects/project_id/regions/region/instanceTemplates/template-id
  """
    instanceGroupManagerName = _messages.StringField(1)
    instanceGroupManagerUri = _messages.StringField(2)
    instanceTemplateName = _messages.StringField(3)
    instanceTemplateUri = _messages.StringField(4)