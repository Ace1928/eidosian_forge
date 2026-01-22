from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class QuotaRule(_messages.Message):
    """`QuotaRule` maps a method to a set of `QuotaGroup`s.

  Fields:
    disableQuota: Indicates if quota checking should be enforced. Quota will
      be disabled for methods without quota rules or with quota rules having
      this field set to true. When this field is set to true, no quota group
      mapping is allowed.
    groups: Quota groups to be used for this method. This supports associating
      a cost with each quota group.
    selector: Selects methods to which this rule applies.  Refer to selector
      for syntax details.
  """
    disableQuota = _messages.BooleanField(1)
    groups = _messages.MessageField('QuotaGroupMapping', 2, repeated=True)
    selector = _messages.StringField(3)