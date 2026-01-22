from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class EdgeConfigstoreBundleBadBundle(_messages.Message):
    """Describes why a bundle is invalid. Intended for use in error details.

  Fields:
    violations: Describes all precondition violations.
  """
    violations = _messages.MessageField('EdgeConfigstoreBundleBadBundleViolation', 1, repeated=True)