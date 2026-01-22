from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DataprocProjectsLocationsBatchesCreateRequest(_messages.Message):
    """A DataprocProjectsLocationsBatchesCreateRequest object.

  Fields:
    batch: A Batch resource to be passed as the request body.
    batchId: Optional. The ID to use for the batch, which will become the
      final component of the batch's resource name.This value must be 4-63
      characters. Valid characters are /[a-z][0-9]-/.
    parent: Required. The parent resource where this batch will be created.
    requestId: Optional. A unique ID used to identify the request. If the
      service receives two CreateBatchRequest (https://cloud.google.com/datapr
      oc/docs/reference/rpc/google.cloud.dataproc.v1#google.cloud.dataproc.v1.
      CreateBatchRequest)s with the same request_id, the second request is
      ignored and the Operation that corresponds to the first Batch created
      and stored in the backend is returned.Recommendation: Set this value to
      a UUID (https://en.wikipedia.org/wiki/Universally_unique_identifier).The
      value must contain only letters (a-z, A-Z), numbers (0-9), underscores
      (_), and hyphens (-). The maximum length is 40 characters.
  """
    batch = _messages.MessageField('Batch', 1)
    batchId = _messages.StringField(2)
    parent = _messages.StringField(3, required=True)
    requestId = _messages.StringField(4)