from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class IamLocationsWorkforcePoolsSubjectsDeleteRequest(_messages.Message):
    """A IamLocationsWorkforcePoolsSubjectsDeleteRequest object.

  Fields:
    name: Required. The resource name of the WorkforcePoolSubject. Special
      characters, like '/' and ':', must be escaped, because all URLs need to
      conform to the "When to Escape and Unescape" section of
      [RFC3986](https://www.ietf.org/rfc/rfc2396.txt). Format: `locations/{loc
      ation}/workforcePools/{workforce_pool_id}/subjects/{subject_id}`
  """
    name = _messages.StringField(1, required=True)