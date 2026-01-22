from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NotebooksProjectsLocationsInstancesIsUpgradeableRequest(_messages.Message):
    """A NotebooksProjectsLocationsInstancesIsUpgradeableRequest object.

  Enums:
    TypeValueValuesEnum: Optional. The optional UpgradeType. Setting this
      field will search for additional compute images to upgrade this
      instance.

  Fields:
    notebookInstance: Required. Format:
      `projects/{project_id}/locations/{location}/instances/{instance_id}`
    type: Optional. The optional UpgradeType. Setting this field will search
      for additional compute images to upgrade this instance.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """Optional. The optional UpgradeType. Setting this field will search for
    additional compute images to upgrade this instance.

    Values:
      UPGRADE_TYPE_UNSPECIFIED: Upgrade type is not specified.
      UPGRADE_FRAMEWORK: Upgrade ML framework.
      UPGRADE_OS: Upgrade Operating System.
      UPGRADE_CUDA: Upgrade CUDA.
      UPGRADE_ALL: Upgrade All (OS, Framework and CUDA).
    """
        UPGRADE_TYPE_UNSPECIFIED = 0
        UPGRADE_FRAMEWORK = 1
        UPGRADE_OS = 2
        UPGRADE_CUDA = 3
        UPGRADE_ALL = 4
    notebookInstance = _messages.StringField(1, required=True)
    type = _messages.EnumField('TypeValueValuesEnum', 2)