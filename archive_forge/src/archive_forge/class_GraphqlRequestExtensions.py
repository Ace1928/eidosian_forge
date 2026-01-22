from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GraphqlRequestExtensions(_messages.Message):
    """GraphqlRequestExtensions contains additional information of
  `GraphqlRequest`.

  Fields:
    impersonate: Optional. If set, impersonate a request with given Firebase
      Auth context and evaluate the auth policies on the operation. If
      omitted, bypass any defined auth policies.
  """
    impersonate = _messages.MessageField('Impersonation', 1)