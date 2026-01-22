from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootRequestResponseTakedownResult(_messages.Message):
    """A LearningGenaiRootRequestResponseTakedownResult object.

  Fields:
    allowed: False when response has to be taken down per above config.
    requestTakedownRegex: Regex used to match the request.
    responseTakedownRegex: Regex used to decide that response should be taken
      down. Empty when response is kept.
  """
    allowed = _messages.BooleanField(1)
    requestTakedownRegex = _messages.StringField(2)
    responseTakedownRegex = _messages.StringField(3)