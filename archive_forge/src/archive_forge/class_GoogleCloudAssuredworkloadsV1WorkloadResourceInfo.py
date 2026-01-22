from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudAssuredworkloadsV1WorkloadResourceInfo(_messages.Message):
    """Represent the resources that are children of this Workload.

  Enums:
    ResourceTypeValueValuesEnum: Indicates the type of resource.

  Fields:
    resourceId: Resource identifier. For a project this represents
      project_number.
    resourceType: Indicates the type of resource.
  """

    class ResourceTypeValueValuesEnum(_messages.Enum):
        """Indicates the type of resource.

    Values:
      RESOURCE_TYPE_UNSPECIFIED: Unknown resource type.
      CONSUMER_PROJECT: Deprecated. Existing workloads will continue to
        support this, but new CreateWorkloadRequests should not specify this
        as an input value.
      CONSUMER_FOLDER: Consumer Folder.
      ENCRYPTION_KEYS_PROJECT: Consumer project containing encryption keys.
      KEYRING: Keyring resource that hosts encryption keys.
    """
        RESOURCE_TYPE_UNSPECIFIED = 0
        CONSUMER_PROJECT = 1
        CONSUMER_FOLDER = 2
        ENCRYPTION_KEYS_PROJECT = 3
        KEYRING = 4
    resourceId = _messages.IntegerField(1)
    resourceType = _messages.EnumField('ResourceTypeValueValuesEnum', 2)