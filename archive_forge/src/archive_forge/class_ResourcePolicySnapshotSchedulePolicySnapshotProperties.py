from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ResourcePolicySnapshotSchedulePolicySnapshotProperties(_messages.Message):
    """Specified snapshot properties for scheduled snapshots created by this
  policy.

  Messages:
    LabelsValue: Labels to apply to scheduled snapshots. These can be later
      modified by the setLabels method. Label values may be empty.

  Fields:
    chainName: Chain name that the snapshot is created in.
    guestFlush: Indication to perform a 'guest aware' snapshot.
    labels: Labels to apply to scheduled snapshots. These can be later
      modified by the setLabels method. Label values may be empty.
    storageLocations: Cloud Storage bucket storage location of the auto
      snapshot (regional or multi-regional).
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Labels to apply to scheduled snapshots. These can be later modified by
    the setLabels method. Label values may be empty.

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
    chainName = _messages.StringField(1)
    guestFlush = _messages.BooleanField(2)
    labels = _messages.MessageField('LabelsValue', 3)
    storageLocations = _messages.StringField(4, repeated=True)