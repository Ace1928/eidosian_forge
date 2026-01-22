from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class NonSdkApiUsageViolationReport(_messages.Message):
    """Contains a summary and examples of non-sdk API usage violations.

  Fields:
    exampleApis: Examples of the detected API usages.
    minSdkVersion: Minimum API level required for the application to run.
    targetSdkVersion: Specifies the API Level on which the application is
      designed to run.
    uniqueApis: Total number of unique Non-SDK API's accessed.
  """
    exampleApis = _messages.MessageField('NonSdkApi', 1, repeated=True)
    minSdkVersion = _messages.IntegerField(2, variant=_messages.Variant.INT32)
    targetSdkVersion = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    uniqueApis = _messages.IntegerField(4, variant=_messages.Variant.INT32)