from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
@encoding.MapUnrecognizedFields('additionalProperties')
class AutoscalingPoliciesValue(_messages.Message):
    """Required. The map with autoscaling policies applied to the cluster.
    The key is the identifier of the policy. It must meet the following
    requirements: * Only contains 1-63 alphanumeric characters and hyphens *
    Begins with an alphabetical character * Ends with a non-hyphen character *
    Not formatted as a UUID * Complies with [RFC
    1034](https://datatracker.ietf.org/doc/html/rfc1034) (section 3.5)
    Currently there map must contain only one element that describes the
    autoscaling policy for compute nodes.

    Messages:
      AdditionalProperty: An additional property for a
        AutoscalingPoliciesValue object.

    Fields:
      additionalProperties: Additional properties of type
        AutoscalingPoliciesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a AutoscalingPoliciesValue object.

      Fields:
        key: Name of the additional property.
        value: A AutoscalingPolicy attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('AutoscalingPolicy', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)