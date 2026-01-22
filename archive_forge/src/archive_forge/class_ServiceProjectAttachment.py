from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServiceProjectAttachment(_messages.Message):
    """ServiceProjectAttachment represents an attachment from a service project
  to a host project. Service projects contain the underlying cloud
  infrastructure resources, and expose these resources to the host project
  through a ServiceProjectAttachment. With the attachments, the host project
  can provide an aggregated view of resources across all service projects.

  Enums:
    StateValueValuesEnum: Output only. ServiceProjectAttachment state.

  Fields:
    createTime: Output only. Create time.
    name: Identifier. The resource name of a ServiceProjectAttachment. Format:
      "projects/{host-project-
      id}/locations/global/serviceProjectAttachments/{service-project-id}."
    serviceProject: Required. Immutable. Service project name in the format:
      "projects/abc" or "projects/123". As input, project name with either
      project id or number are accepted. As output, this field will contain
      project number.
    state: Output only. ServiceProjectAttachment state.
    uid: Output only. A globally unique identifier (in UUID4 format) for the
      `ServiceProjectAttachment`.
  """

    class StateValueValuesEnum(_messages.Enum):
        """Output only. ServiceProjectAttachment state.

    Values:
      STATE_UNSPECIFIED: Unspecified state.
      CREATING: The ServiceProjectAttachment is being created.
      ACTIVE: The ServiceProjectAttachment is ready. This means Services and
        Workloads under the corresponding ServiceProjectAttachment is ready
        for registration.
      DELETING: The ServiceProjectAttachment is being deleted.
    """
        STATE_UNSPECIFIED = 0
        CREATING = 1
        ACTIVE = 2
        DELETING = 3
    createTime = _messages.StringField(1)
    name = _messages.StringField(2)
    serviceProject = _messages.StringField(3)
    state = _messages.EnumField('StateValueValuesEnum', 4)
    uid = _messages.StringField(5)