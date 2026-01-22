from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Animation(_messages.Message):
    """Animation types.

  Fields:
    animationEnd: End previous animation.
    animationFade: Display overlay object with fade animation.
    animationStatic: Display static overlay object.
  """
    animationEnd = _messages.MessageField('AnimationEnd', 1)
    animationFade = _messages.MessageField('AnimationFade', 2)
    animationStatic = _messages.MessageField('AnimationStatic', 3)