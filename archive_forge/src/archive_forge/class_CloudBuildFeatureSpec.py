from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CloudBuildFeatureSpec(_messages.Message):
    """Cloud Build for Anthos feature spec. This is required since Feature
  proto requires a spec.

  Messages:
    MembershipConfigsValue: The map from membership path (e.g. projects/foo-
      proj/locations/global/memberships/bar) to the CloudBuildMembershipConfig
      that is chosen for that member cluster. If CloudBuild feature is enabled
      for a hub and the membership path of a cluster in that hub exists in
      this map then it has Cloud Build hub feature enabled for that particular
      cluster.

  Fields:
    membershipConfigs: The map from membership path (e.g. projects/foo-
      proj/locations/global/memberships/bar) to the CloudBuildMembershipConfig
      that is chosen for that member cluster. If CloudBuild feature is enabled
      for a hub and the membership path of a cluster in that hub exists in
      this map then it has Cloud Build hub feature enabled for that particular
      cluster.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class MembershipConfigsValue(_messages.Message):
        """The map from membership path (e.g. projects/foo-
    proj/locations/global/memberships/bar) to the CloudBuildMembershipConfig
    that is chosen for that member cluster. If CloudBuild feature is enabled
    for a hub and the membership path of a cluster in that hub exists in this
    map then it has Cloud Build hub feature enabled for that particular
    cluster.

    Messages:
      AdditionalProperty: An additional property for a MembershipConfigsValue
        object.

    Fields:
      additionalProperties: Additional properties of type
        MembershipConfigsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a MembershipConfigsValue object.

      Fields:
        key: Name of the additional property.
        value: A CloudBuildMembershipConfig attribute.
      """
            key = _messages.StringField(1)
            value = _messages.MessageField('CloudBuildMembershipConfig', 2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    membershipConfigs = _messages.MessageField('MembershipConfigsValue', 1)