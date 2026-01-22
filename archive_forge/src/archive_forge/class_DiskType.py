from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DiskType(_messages.Message):
    """Represents a Disk Type resource. Google Compute Engine has two Disk Type
  resources: * [Regional](/compute/docs/reference/rest/beta/regionDiskTypes) *
  [Zonal](/compute/docs/reference/rest/beta/diskTypes) You can choose from a
  variety of disk types based on your needs. For more information, read
  Storage options. The diskTypes resource represents disk types for a zonal
  persistent disk. For more information, read Zonal persistent disks. The
  regionDiskTypes resource represents disk types for a regional persistent
  disk. For more information, read Regional persistent disks.

  Fields:
    creationTimestamp: [Output Only] Creation timestamp in RFC3339 text
      format.
    defaultDiskSizeGb: [Output Only] Server-defined default disk size in GB.
    deprecated: [Output Only] The deprecation status associated with this disk
      type.
    description: [Output Only] An optional description of this resource.
    id: [Output Only] The unique identifier for the resource. This identifier
      is defined by the server.
    kind: [Output Only] Type of the resource. Always compute#diskType for disk
      types.
    name: [Output Only] Name of the resource.
    region: [Output Only] URL of the region where the disk type resides. Only
      applicable for regional resources. You must specify this field as part
      of the HTTP request URL. It is not settable as a field in the request
      body.
    selfLink: [Output Only] Server-defined URL for the resource.
    validDiskSize: [Output Only] An optional textual description of the valid
      disk size, such as "10GB-10TB".
    zone: [Output Only] URL of the zone where the disk type resides. You must
      specify this field as part of the HTTP request URL. It is not settable
      as a field in the request body.
  """
    creationTimestamp = _messages.StringField(1)
    defaultDiskSizeGb = _messages.IntegerField(2)
    deprecated = _messages.MessageField('DeprecationStatus', 3)
    description = _messages.StringField(4)
    id = _messages.IntegerField(5, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(6, default='compute#diskType')
    name = _messages.StringField(7)
    region = _messages.StringField(8)
    selfLink = _messages.StringField(9)
    validDiskSize = _messages.StringField(10)
    zone = _messages.StringField(11)