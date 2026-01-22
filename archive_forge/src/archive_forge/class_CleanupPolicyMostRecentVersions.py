from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class CleanupPolicyMostRecentVersions(_messages.Message):
    """CleanupPolicyMostRecentVersions is an alternate condition of a
  CleanupPolicy for retaining a minimum number of versions.

  Fields:
    keepCount: Minimum number of versions to keep.
    packageNamePrefixes: List of package name prefixes that will apply this
      rule.
  """
    keepCount = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    packageNamePrefixes = _messages.StringField(2, repeated=True)