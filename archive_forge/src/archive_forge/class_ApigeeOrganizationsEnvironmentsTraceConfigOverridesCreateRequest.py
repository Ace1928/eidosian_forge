from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ApigeeOrganizationsEnvironmentsTraceConfigOverridesCreateRequest(_messages.Message):
    """A ApigeeOrganizationsEnvironmentsTraceConfigOverridesCreateRequest
  object.

  Fields:
    googleCloudApigeeV1TraceConfigOverride: A
      GoogleCloudApigeeV1TraceConfigOverride resource to be passed as the
      request body.
    parent: Required. Parent resource of the trace configuration override. Use
      the following structure in your request.
      "organizations/*/environments/*/traceConfig".
  """
    googleCloudApigeeV1TraceConfigOverride = _messages.MessageField('GoogleCloudApigeeV1TraceConfigOverride', 1)
    parent = _messages.StringField(2, required=True)