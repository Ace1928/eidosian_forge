from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class InstanceTemplate(_messages.Message):
    """Represents an Instance Template resource. Google Compute Engine has two
  Instance Template resources: *
  [Global](/compute/docs/reference/rest/beta/instanceTemplates) *
  [Regional](/compute/docs/reference/rest/beta/regionInstanceTemplates) You
  can reuse a global instance template in different regions whereas you can
  use a regional instance template in a specified region only. If you want to
  reduce cross-region dependency or achieve data residency, use a regional
  instance template. To create VMs, managed instance groups, and reservations,
  you can use either global or regional instance templates. For more
  information, read Instance Templates.

  Fields:
    creationTimestamp: [Output Only] The creation timestamp for this instance
      template in RFC3339 text format.
    description: An optional description of this resource. Provide this
      property when you create the resource.
    id: [Output Only] A unique identifier for this instance template. The
      server defines this identifier.
    kind: [Output Only] The resource type, which is always
      compute#instanceTemplate for instance templates.
    name: Name of the resource; provided by the client when the resource is
      created. The name must be 1-63 characters long, and comply with RFC1035.
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z]([-a-z0-9]*[a-z0-9])?` which means the first
      character must be a lowercase letter, and all following characters must
      be a dash, lowercase letter, or digit, except the last character, which
      cannot be a dash.
    properties: The instance properties for this instance template.
    region: [Output Only] URL of the region where the instance template
      resides. Only applicable for regional resources.
    selfLink: [Output Only] The URL for this instance template. The server
      defines this URL.
    sourceInstance: The source instance used to create the template. You can
      provide this as a partial or full URL to the resource. For example, the
      following are valid values: -
      https://www.googleapis.com/compute/v1/projects/project/zones/zone
      /instances/instance - projects/project/zones/zone/instances/instance
    sourceInstanceParams: The source instance params to use to create this
      instance template.
  """
    creationTimestamp = _messages.StringField(1)
    description = _messages.StringField(2)
    id = _messages.IntegerField(3, variant=_messages.Variant.UINT64)
    kind = _messages.StringField(4, default='compute#instanceTemplate')
    name = _messages.StringField(5)
    properties = _messages.MessageField('InstanceProperties', 6)
    region = _messages.StringField(7)
    selfLink = _messages.StringField(8)
    sourceInstance = _messages.StringField(9)
    sourceInstanceParams = _messages.MessageField('SourceInstanceParams', 10)