from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsRegionsClustersNodeGroupsCreateRequest(_messages.Message):
    """A DataprocProjectsRegionsClustersNodeGroupsCreateRequest object.

  Fields:
    nodeGroup: A NodeGroup resource to be passed as the request body.
    nodeGroupId: Optional. An optional node group ID. Generated if not
      specified.The ID must contain only letters (a-z, A-Z), numbers (0-9),
      underscores (_), and hyphens (-). Cannot begin or end with underscore or
      hyphen. Must consist of from 3 to 33 characters.
    parent: Required. The parent resource where this node group will be
      created. Format: projects/{project}/regions/{region}/clusters/{cluster}
    parentOperationId: Optional. operation id of the parent operation sending
      the create request
    requestId: Optional. A unique ID used to identify the request. If the
      server receives two CreateNodeGroupRequest (https://cloud.google.com/dat
      aproc/docs/reference/rpc/google.cloud.dataproc.v1#google.cloud.dataproc.
      v1.CreateNodeGroupRequest) with the same ID, the second request is
      ignored and the first google.longrunning.Operation created and stored in
      the backend is returned.Recommendation: Set this value to a UUID
      (https://en.wikipedia.org/wiki/Universally_unique_identifier).The ID
      must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """
    nodeGroup = _messages.MessageField('NodeGroup', 1)
    nodeGroupId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    parentOperationId = _messages.StringField(4)
    requestId = _messages.StringField(5)