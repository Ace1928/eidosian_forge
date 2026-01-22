from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class AlloydbProjectsLocationsClustersInstancesPatchRequest(_messages.Message):
    """A AlloydbProjectsLocationsClustersInstancesPatchRequest object.

  Fields:
    allowMissing: Optional. If set to true, update succeeds even if instance
      is not found. In that case, a new instance is created and `update_mask`
      is ignored.
    instance: A Instance resource to be passed as the request body.
    name: Output only. The name of the instance resource with the format: * pr
      ojects/{project}/locations/{region}/clusters/{cluster_id}/instances/{ins
      tance_id} where the cluster and instance ID segments should satisfy the
      regex expression `[a-z]([a-z0-9-]{0,61}[a-z0-9])?`, e.g. 1-63 characters
      of lowercase letters, numbers, and dashes, starting with a letter, and
      ending with a letter or number. For more details see
      https://google.aip.dev/122. The prefix of the instance resource name is
      the name of the parent resource: *
      projects/{project}/locations/{region}/clusters/{cluster_id}
    requestId: Optional. An optional request ID to identify requests. Specify
      a unique request ID so that if you must retry your request, the server
      will know to ignore the request if it has already been completed. The
      server will guarantee that for at least 60 minutes since the first
      request. For example, consider a situation where you make an initial
      request and the request times out. If you make the request again with
      the same request ID, the server can check if original operation with the
      same request ID was received, and if so, will ignore the second request.
      This prevents clients from accidentally creating duplicate commitments.
      The request ID must be a valid UUID with the exception that zero UUID is
      not supported (00000000-0000-0000-0000-000000000000).
    updateMask: Optional. Field mask is used to specify the fields to be
      overwritten in the Instance resource by the update. The fields specified
      in the update_mask are relative to the resource, not the full request. A
      field will be overwritten if it is in the mask. If the user does not
      provide a mask then all fields will be overwritten.
    validateOnly: Optional. If set, performs request validation (e.g.
      permission checks and any other type of validation), but do not actually
      execute the update request.
  """
    allowMissing = _messages.BooleanField(1)
    instance = _messages.MessageField('Instance', 2)
    name = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)
    updateMask = _messages.StringField(5)
    validateOnly = _messages.BooleanField(6)