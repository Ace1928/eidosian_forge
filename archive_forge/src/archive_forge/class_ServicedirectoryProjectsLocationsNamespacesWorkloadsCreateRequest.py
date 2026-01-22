from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class ServicedirectoryProjectsLocationsNamespacesWorkloadsCreateRequest(_messages.Message):
    """A ServicedirectoryProjectsLocationsNamespacesWorkloadsCreateRequest
  object.

  Fields:
    parent: Required. The resource name of the namespace this service workload
      will belong to.
    workload: A Workload resource to be passed as the request body.
    workloadId: Required. The Resource ID must be 1-63 characters long, and
      comply with [RFC 1035](https://www.ietf.org/rfc/rfc1035.txt).
      Specifically, the name must be 1-63 characters long and match the
      regular expression `[a-z](?:[-a-z0-9]{0,61}[a-z0-9])?` which means the
      first character must be a lowercase letter, and all following characters
      must be a dash, lowercase letter, or digit, except the last character,
      which cannot be a dash.
  """
    parent = _messages.StringField(1, required=True)
    workload = _messages.MessageField('Workload', 2)
    workloadId = _messages.StringField(3)