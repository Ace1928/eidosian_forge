from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class LakeResourcesValue(_messages.Message):
    """A reference to the Lake resources that this metastore service is
    attached to. The key is the lake resource name. Example:
    projects/{project_number}/locations/{location_id}/lakes/{lake_id}.

    Messages:
      AdditionalProperty: An additional property for a LakeResourcesValue
        object.

    Fields:
      additionalProperties: Additional properties of type LakeResourcesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a LakeResourcesValue object.

      Fields:
        key: Name of the additional property.
        value: A Lake attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('Lake', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)