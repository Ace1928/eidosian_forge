from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class TestTiming(_messages.Message):
    """Testing timing break down to know phases.

  Fields:
    testProcessDuration: How long it took to run the test process. - In
      response: present if previously set. - In create/update request:
      optional
  """
    testProcessDuration = _messages.MessageField('Duration', 1)