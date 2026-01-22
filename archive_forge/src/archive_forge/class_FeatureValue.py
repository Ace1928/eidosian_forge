from __future__ import absolute_import
from apitools.base.protorpclite import message_types as _message_types
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class FeatureValue(_messages.Message):
    """Representative value of a single feature within the cluster.

  Fields:
    categoricalValue: The categorical feature value.
    featureColumn: The feature column name.
    numericalValue: The numerical feature value. This is the centroid value
      for this feature.
  """
    categoricalValue = _messages.MessageField('CategoricalValue', 1)
    featureColumn = _messages.StringField(2)
    numericalValue = _messages.FloatField(3)