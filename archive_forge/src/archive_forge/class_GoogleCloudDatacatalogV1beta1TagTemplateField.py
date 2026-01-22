from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDatacatalogV1beta1TagTemplateField(_messages.Message):
    """The template for an individual field within a tag template.

  Fields:
    description: The description for this field. Defaults to an empty string.
    displayName: The display name for this field. Defaults to an empty string.
    isRequired: Whether this is a required field. Defaults to false.
    name: Output only. Identifier. The resource name of the tag template field
      in URL format. Example: * projects/{project_id}/locations/{location}/tag
      Templates/{tag_template}/fields/{field} Note that this TagTemplateField
      may not actually be stored in the location in this name.
    order: The order of this field with respect to other fields in this tag
      template. A higher value indicates a more important field. The value can
      be negative. Multiple fields can have the same order, and field orders
      within a tag do not have to be sequential.
    type: Required. The type of value this tag field can contain.
  """
    description = _messages.StringField(1)
    displayName = _messages.StringField(2)
    isRequired = _messages.BooleanField(3)
    name = _messages.StringField(4)
    order = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    type = _messages.MessageField('GoogleCloudDatacatalogV1beta1FieldType', 6)