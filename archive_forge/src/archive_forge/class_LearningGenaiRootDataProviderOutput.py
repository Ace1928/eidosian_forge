from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootDataProviderOutput(_messages.Message):
    """A LearningGenaiRootDataProviderOutput object.

  Fields:
    name: A string attribute.
    status: If set, this DataProvider failed and this is the error message.
  """
    name = _messages.StringField(1)
    status = _messages.MessageField('UtilStatusProto', 2)