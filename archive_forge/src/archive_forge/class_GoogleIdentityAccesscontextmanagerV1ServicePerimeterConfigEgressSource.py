from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleIdentityAccesscontextmanagerV1ServicePerimeterConfigEgressSource(_messages.Message):
    """The source that EgressPolicy authorizes access from inside the
  ServicePerimeter to somewhere outside the ServicePerimeter boundaries.

  Fields:
    accessLevel: An AccessLevel resource name that allows protected resources
      inside the ServicePerimeters to access outside the ServicePerimeter
      boundaries. AccessLevels listed must be in the same policy as this
      ServicePerimeter. Referencing a nonexistent AccessLevel will cause an
      error. If an AccessLevel name is not specified, only resources within
      the perimeter can be accessed through Google Cloud calls with request
      origins within the perimeter. Example:
      `accessPolicies/MY_POLICY/accessLevels/MY_LEVEL`. If a single `*` is
      specified for `access_level`, then all EgressSources will be allowed.
    resource: A Google Cloud resource that is allowed to egress the perimeter.
      Requests from these resources are allowed to access data outside the.
      perimeter. Only projects and VPCs are allowed. Project format:
      `projects/{project_number}`. VPC format: `//compute.googleapis.com/proje
      cts/{PROJECT_ID}/global/networks/{NETWORK_NAME}`. The resource might be
      in any Google Cloud organization, not just the organization that the
      perimeter is defined in. `*` is not allowed, the case of allowing all
      Google Cloud resources only is not supported.
  """
    accessLevel = _messages.StringField(1)
    resource = _messages.StringField(2)