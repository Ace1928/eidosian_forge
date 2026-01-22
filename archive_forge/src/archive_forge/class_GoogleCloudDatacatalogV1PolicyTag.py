from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1PolicyTag(_messages.Message):
    """Denotes one policy tag in a taxonomy, for example, SSN. Policy tags can
  be defined in a hierarchy. For example: ``` + Geolocation + LatLong + City +
  ZipCode ``` Where the "Geolocation" policy tag contains three children.

  Fields:
    childPolicyTags: Output only. Resource names of child policy tags of this
      policy tag.
    description: Description of this policy tag. If not set, defaults to
      empty. The description must contain only Unicode characters, tabs,
      newlines, carriage returns and page breaks, and be at most 2000 bytes
      long when encoded in UTF-8.
    displayName: Required. User-defined name of this policy tag. The name
      can't start or end with spaces and must be unique within the parent
      taxonomy, contain only Unicode letters, numbers, underscores, dashes and
      spaces, and be at most 200 bytes long when encoded in UTF-8.
    name: Identifier. Resource name of this policy tag in the URL format. The
      policy tag manager generates unique taxonomy IDs and policy tag IDs.
    parentPolicyTag: Resource name of this policy tag's parent policy tag. If
      empty, this is a top level tag. If not set, defaults to an empty string.
      For example, for the "LatLong" policy tag in the example above, this
      field contains the resource name of the "Geolocation" policy tag, and,
      for "Geolocation", this field is empty.
  """
    childPolicyTags = _messages.StringField(1, repeated=True)
    description = _messages.StringField(2)
    displayName = _messages.StringField(3)
    name = _messages.StringField(4)
    parentPolicyTag = _messages.StringField(5)