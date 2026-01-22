from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsTraceConfigOverridesListRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsTraceConfigOverridesListRequest object.

  Fields:
    pageSize: Maximum number of trace configuration overrides to return. If
      not specified, the maximum number returned is 25. The maximum number
      cannot exceed 100.
    pageToken: A page token, returned from a previous
      `ListTraceConfigOverrides` call. Token value that can be used to
      retrieve the subsequent page. When paginating, all other parameters
      provided to `ListTraceConfigOverrides` must match those specified in the
      call to obtain the page token.
    parent: Required. Parent resource of the trace configuration override. Use
      the following structure in your request:
      "organizations/*/environments/*/traceConfig".
  """
    pageSize = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    pageToken = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)