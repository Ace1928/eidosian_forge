from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
@encoding.MapUnrecognizedFields('additionalProperties')
class ClusterAdmissionRulesValue(_messages.Message):
    """Optional. Per-cluster admission rules. Cluster spec format:
    `location.clusterId`. There can be at most one admission rule per cluster
    spec. A `location` is either a compute zone (e.g. us-central1-a) or a
    region (e.g. us-central1). For `clusterId` syntax restrictions see
    https://cloud.google.com/container-
    engine/reference/rest/v1/projects.zones.clusters.

    Messages:
      AdditionalProperty: An additional property for a
        ClusterAdmissionRulesValue object.

    Fields:
      additionalProperties: Additional properties of type
        ClusterAdmissionRulesValue
    """

    class AdditionalProperty(_messages.Message):
        """An additional property for a ClusterAdmissionRulesValue object.

      Fields:
        key: Name of the additional property.
        value: A AdmissionRule attribute.
      """
        key = _messages.StringField(1)
        value = _messages.MessageField('AdmissionRule', 2)
    additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)