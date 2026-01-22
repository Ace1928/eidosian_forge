from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ManagedCluster(_messages.Message):
    """Cluster that is managed by the workflow.

  Messages:
    LabelsValue: Optional. The labels to associate with this cluster.Label
      keys must be between 1 and 63 characters long, and must conform to the
      following PCRE regular expression: \\p{Ll}\\p{Lo}{0,62}Label values must
      be between 1 and 63 characters long, and must conform to the following
      PCRE regular expression: \\p{Ll}\\p{Lo}\\p{N}_-{0,63}No more than 32 labels
      can be associated with a given cluster.

  Fields:
    clusterName: Required. The cluster name prefix. A unique cluster name will
      be formed by appending a random suffix.The name must contain only lower-
      case letters (a-z), numbers (0-9), and hyphens (-). Must begin with a
      letter. Cannot begin or end with hyphen. Must consist of between 2 and
      35 characters.
    config: Required. The cluster configuration.
    labels: Optional. The labels to associate with this cluster.Label keys
      must be between 1 and 63 characters long, and must conform to the
      following PCRE regular expression: \\p{Ll}\\p{Lo}{0,62}Label values must
      be between 1 and 63 characters long, and must conform to the following
      PCRE regular expression: \\p{Ll}\\p{Lo}\\p{N}_-{0,63}No more than 32 labels
      can be associated with a given cluster.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Optional. The labels to associate with this cluster.Label keys must be
    between 1 and 63 characters long, and must conform to the following PCRE
    regular expression: \\p{Ll}\\p{Lo}{0,62}Label values must be between 1 and
    63 characters long, and must conform to the following PCRE regular
    expression: \\p{Ll}\\p{Lo}\\p{N}_-{0,63}No more than 32 labels can be
    associated with a given cluster.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    clusterName = _messages.StringField(1)
    config = _messages.MessageField('ClusterConfig', 2)
    labels = _messages.MessageField('LabelsValue', 3)