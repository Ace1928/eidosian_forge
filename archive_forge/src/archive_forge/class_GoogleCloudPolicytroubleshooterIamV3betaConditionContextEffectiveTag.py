from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudPolicytroubleshooterIamV3betaConditionContextEffectiveTag(_messages.Message):
    """A tag that applies to a resource during policy evaluation. Tags can be
  either directly bound to a resource or inherited from its ancestor.
  `EffectiveTag` contains the `name` and `namespaced_name` of the tag value
  and tag key, with additional fields of `inherited` to indicate the
  inheritance status of the effective tag.

  Fields:
    inherited: Output only. Indicates the inheritance status of a tag value
      attached to the given resource. If the tag value is inherited from one
      of the resource's ancestors, inherited will be true. If false, then the
      tag value is directly attached to the resource, inherited will be false.
    namespacedTagKey: Output only. The namespaced name of the TagKey. Can be
      in the form `{organization_id}/{tag_key_short_name}` or
      `{project_id}/{tag_key_short_name}` or
      `{project_number}/{tag_key_short_name}`.
    namespacedTagValue: Output only. The namespaced name of the TagValue. Can
      be in the form
      `{organization_id}/{tag_key_short_name}/{tag_value_short_name}` or
      `{project_id}/{tag_key_short_name}/{tag_value_short_name}` or
      `{project_number}/{tag_key_short_name}/{tag_value_short_name}`.
    tagKey: Output only. The name of the TagKey, in the format `tagKeys/{id}`,
      such as `tagKeys/123`.
    tagKeyParentName: The parent name of the tag key. Must be in the format
      `organizations/{organization_id}` or `projects/{project_number}`
    tagValue: Output only. Resource name for TagValue in the format
      `tagValues/456`.
  """
    inherited = _messages.BooleanField(1)
    namespacedTagKey = _messages.StringField(2)
    namespacedTagValue = _messages.StringField(3)
    tagKey = _messages.StringField(4)
    tagKeyParentName = _messages.StringField(5)
    tagValue = _messages.StringField(6)