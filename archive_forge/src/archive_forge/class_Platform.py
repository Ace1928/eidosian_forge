from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class Platform(_messages.Message):
    """A platform supported by Binary Authorization platform policy.

  Fields:
    name: Output only. The relative resource name of the platform supported by
      Binary Authorization platform policies, in the form of
      `projects/*/platforms/*`.
  """
    name = _messages.StringField(1)