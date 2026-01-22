from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class WorkloadSource(_messages.Message):
    """Defines which workloads can attest an identity within a pool. When a
  WorkloadSource is defined under a namespace, matching workloads may receive
  any identity within that namespace. When a WorkloadSource is defined under a
  managed identity, matching workloads may receive that specific identity.
  Each WorkloadSource may set at most 50 workload selectors.

  Fields:
    etag: Optional. The etag for this resource. If this is provided on update,
      it must match the server's etag.
    identityAssignments: Optional. Defines how a matched workload has its
      identity assigned. This option may only be set when the Workload Source
      is defined on a Namespace.
    name: Output only. The resource name of the workload source. If ID of the
      WorkloadSource resource determines which workloads may be matched. The
      following formats are supported: - `project-{project_number}` matches
      workloads within the referenced Google Cloud project.
    singleAttributeSelectors: Optional. Defines the set of attributes that a
      workload must attest in order to be matched by the policy.
  """
    etag = _messages.StringField(1)
    identityAssignments = _messages.MessageField('IdentityAssignment', 2, repeated=True)
    name = _messages.StringField(3)
    singleAttributeSelectors = _messages.MessageField('SingleAttributeSelector', 4, repeated=True)