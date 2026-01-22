from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StagesSummary(_messages.Message):
    """Data related to Stages page summary

  Fields:
    applicationId: A string attribute.
    numActiveStages: A integer attribute.
    numCompletedStages: A integer attribute.
    numFailedStages: A integer attribute.
    numPendingStages: A integer attribute.
    numSkippedStages: A integer attribute.
  """
    applicationId = _messages.StringField(1)
    numActiveStages = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    numCompletedStages = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    numFailedStages = _messages.IntegerField(4, variant=_messages.Variant.INT32)
    numPendingStages = _messages.IntegerField(5, variant=_messages.Variant.INT32)
    numSkippedStages = _messages.IntegerField(6, variant=_messages.Variant.INT32)