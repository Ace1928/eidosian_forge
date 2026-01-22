from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AccessSparkApplicationEnvironmentInfoResponse(_messages.Message):
    """Environment details of a Saprk Application.

  Fields:
    applicationEnvironmentInfo: Details about the Environment that the
      application is running in.
  """
    applicationEnvironmentInfo = _messages.MessageField('ApplicationEnvironmentInfo', 1)