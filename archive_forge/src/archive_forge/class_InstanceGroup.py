from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceGroup(_messages.Message):
    """Represents an Instance Group resource. Instance Groups can be used to
  configure a target for load balancing. Instance groups can either be managed
  or unmanaged. To create managed instance groups, use the
  instanceGroupManager or regionInstanceGroupManager resource instead. Use
  zonal unmanaged instance groups if you need to apply load balancing to
  groups of heterogeneous instances or if you need to manage the instances
  yourself. You cannot create regional unmanaged instance groups. For more
  information, read Instance groups.

  Fields:
    creationTimestamp: [Output Only] The creation timestamp for this instance
      group in RFC3339 text format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    fingerprint: [Output Only] The fingerprint of the named ports. The system
      uses this fingerprint to detect conflicts when multiple users change the
      named ports concurrently.
    id: [Output Only] A unique identifier for this instance group, generated
      by the server.
    kind: [Output Only] The resource type, which is always
      compute#instanceGroup for instance groups.
    name: The name of the instance group. The name must be 1-63 characters
      long, and comply with RFC1035.
    namedPorts:  Assigns a name to a port number. For example: {name: "http",
      port: 80} This allows the system to reference ports by the assigned name
      instead of a port number. Named ports can also contain multiple ports.
      For example: [{name: "app1", port: 8080}, {name: "app1", port: 8081},
      {name: "app2", port: 8082}] Named ports apply to all instances in this
      instance group.
    network: [Output Only] The URL of the network to which all instances in
      the instance group belong. If your instance has multiple network
      interfaces, then the network and subnetwork fields only refer to the
      network and subnet used by your primary interface (nic0).
    region: [Output Only] The URL of the region where the instance group is
      located (for regional resources).
    selfLink: [Output Only] The URL for this instance group. The server
      generates this URL.
    size: [Output Only] The total number of instances in the instance group.
    subnetwork: [Output Only] The URL of the subnetwork to which all instances
      in the instance group belong. If your instance has multiple network
      interfaces, then the network and subnetwork fields only refer to the
      network and subnet used by your primary interface (nic0).
    zone: [Output Only] The URL of the zone where the instance group is
      located (for zonal resources).
  """
    creationTimestamp = _messages.StringField(1)
    description = _messages.StringField(2)
    fingerprint = _messages.BytesField(3)
    id = _messages.IntegerField(4, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(5, default='compute#instanceGroup')
    name = _messages.StringField(6)
    namedPorts = _messages.MessageField('NamedPort', 7, repeated=True)
    network = _messages.StringField(8)
    region = _messages.StringField(9)
    selfLink = _messages.StringField(10)
    size = _messages.IntegerField(11, variant=_messages.Variant.INT32)
    subnetwork = _messages.StringField(12)
    zone = _messages.StringField(13)