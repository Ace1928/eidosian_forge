from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataBoostReadLocalWrites(_messages.Message):
    """Checks that all writes before the consistency token was generated in the
  same cluster are readable by Databoost.
  """