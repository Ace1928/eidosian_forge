from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroupManagerResizeRequest(_messages.Message):
    """InstanceGroupManagerResizeRequest represents a request to create a
  number of VMs: either immediately or by queuing the request for the
  specified time. This resize request is nested under InstanceGroupManager and
  the VMs created by this request are added to the owning
  InstanceGroupManager.

  Enums:
    StateValueValuesEnum: [Output only] Current state of the request.

  Fields:
    count: This field is deprecated, please use resize_by instead. The count
      of instances to create as part of this resize request.
    creationTimestamp: [Output Only] The creation timestamp for this resize
      request in RFC3339 text format.
    description: An optional description of this resource.
    id: [Output Only] A unique identifier for this resource type. The server
      generates this identifier.
    kind: [Output Only] The resource type, which is always
      compute#instanceGroupManagerResizeRequest for resize requests.
    name: The name of this resize request. The name must be 1-63 characters
      long, and comply with RFC1035.
    requestedRunDuration: Requested run duration for instances that will be
      created by this request. At the end of the run duration instance will be
      deleted.
    resizeBy: The number of instances to be created by this resize request.
      The group's target size will be increased by this number.
    selfLink: [Output Only] The URL for this resize request. The server
      defines this URL.
    selfLinkWithId: [Output Only] Server-defined URL for this resource with
      the resource id.
    state: [Output only] Current state of the request.
    status: [Output only] Status of the request.
    zone: [Output Only] The URL of a zone where the resize request is located.
      Populated only for zonal resize requests.
  """

    class StateValueValuesEnum(_messages.Enum):
        """[Output only] Current state of the request.

    Values:
      ACCEPTED: The request was created successfully and was accepted for
        provisioning when the capacity becomes available.
      CANCELLED: The request is cancelled.
      CREATING: Resize request is being created and may still fail creation.
      FAILED: The request failed before or during provisioning. If the request
        fails during provisioning, any VMs that were created during
        provisioning are rolled back and removed from the MIG.
      PROVISIONING: The value is deprecated. ResizeRequests would stay in the
        ACCEPTED state during provisioning attempts. The target resource(s)
        are being provisioned.
      STATE_UNSPECIFIED: Default value. This value should never be returned.
      SUCCEEDED: The request succeeded.
    """
        ACCEPTED = 0
        CANCELLED = 1
        CREATING = 2
        FAILED = 3
        PROVISIONING = 4
        STATE_UNSPECIFIED = 5
        SUCCEEDED = 6
    count = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    creationTimestamp = _messages.StringField(2)
    description = _messages.StringField(3)
    id = _messages.IntegerField(4, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(5, default='compute#instanceGroupManagerResizeRequest')
    name = _messages.StringField(6)
    requestedRunDuration = _messages.MessageField('Duration', 7)
    resizeBy = _messages.IntegerField(8, variant=_messages.Variant.INT32)
    selfLink = _messages.StringField(9)
    selfLinkWithId = _messages.StringField(10)
    state = _messages.EnumField('StateValueValuesEnum', 11)
    status = _messages.MessageField('InstanceGroupManagerResizeRequestStatus', 12)
    zone = _messages.StringField(13)