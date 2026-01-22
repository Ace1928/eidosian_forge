from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class LearningGenaiRootRAIOutput(_messages.Message):
    """This is per harm.

  Fields:
    allowed: A boolean attribute.
    harm: A LearningGenaiRootHarm attribute.
    name: A string attribute.
    score: A number attribute.
  """
    allowed = _messages.BooleanField(1)
    harm = _messages.MessageField('LearningGenaiRootHarm', 2)
    name = _messages.StringField(3)
    score = _messages.FloatField(4)