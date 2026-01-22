from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class PathNodeAssociatedFinding(_messages.Message):
    """A finding that is associated with this node in the attack path.

  Fields:
    canonicalFinding: Canonical name of the associated findings. Example:
      organizations/123/sources/456/findings/789
    findingCategory: The additional taxonomy group within findings from a
      given source.
    name: Full resource name of the finding.
  """
    canonicalFinding = _messages.StringField(1)
    findingCategory = _messages.StringField(2)
    name = _messages.StringField(3)