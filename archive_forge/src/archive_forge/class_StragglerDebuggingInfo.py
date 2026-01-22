from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class StragglerDebuggingInfo(_messages.Message):
    """Information useful for debugging a straggler. Each type will provide
  specialized debugging information relevant for a particular cause. The
  StragglerDebuggingInfo will be 1:1 mapping to the StragglerCause enum.

  Fields:
    hotKey: Hot key debugging details.
  """
    hotKey = _messages.MessageField('HotKeyDebuggingInfo', 1)