from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class UpdateLabelsNodeGroupRequest(_messages.Message):
    """A request to update the labels of a node group.

  Messages:
    LabelsValue: Required. The labels to associate with this Node Group. Label
      keys must contain 1 to 63 characters, and must conform to RFC 1035
      (https://www.ietf.org/rfc/rfc1035.txt). Label values may be empty, but,
      if present, must contain 1 to 63 characters, and must conform to RFC
      1035 (https://www.ietf.org/rfc/rfc1035.txt). No more than 32 labels can
      be associated with a cluster.

  Fields:
    labels: Required. The labels to associate with this Node Group. Label keys
      must contain 1 to 63 characters, and must conform to RFC 1035
      (https://www.ietf.org/rfc/rfc1035.txt). Label values may be empty, but,
      if present, must contain 1 to 63 characters, and must conform to RFC
      1035 (https://www.ietf.org/rfc/rfc1035.txt). No more than 32 labels can
      be associated with a cluster.
    parentOperationId: Optional. Operation id of the parent operation sending
      the update labels request.
    requestId: Optional. A unique ID used to identify the request. If the
      server receives two UpdateLabelsNodeGroupRequest (https://cloud.google.c
      om/dataproc/docs/reference/rpc/google.cloud.dataproc.v1#google.cloud.dat
      aproc.v1.UpdateLabelsNodeGroupRequests) with the same ID, the second
      request is ignored and the first google.longrunning.Operation created
      and stored in the backend is returned.Recommendation: Set this value to
      a UUID (https://en.wikipedia.org/wiki/Universally_unique_identifier).The
      ID must contain only letters (a-z, A-Z), numbers (0-9), underscores (_),
      and hyphens (-). The maximum length is 40 characters.
  """

    @encoding.MapUnrecognizedFields('additionalProperties')
    class LabelsValue(_messages.Message):
        """Required. The labels to associate with this Node Group. Label keys
    must contain 1 to 63 characters, and must conform to RFC 1035
    (https://www.ietf.org/rfc/rfc1035.txt). Label values may be empty, but, if
    present, must contain 1 to 63 characters, and must conform to RFC 1035
    (https://www.ietf.org/rfc/rfc1035.txt). No more than 32 labels can be
    associated with a cluster.

    Messages:
      AdditionalProperty: An additional property for a LabelsValue object.

    Fields:
      additionalProperties: Additional properties of type LabelsValue
    """

        class AdditionalProperty(_messages.Message):
            """An additional property for a LabelsValue object.

      Fields:
        key: Name of the additional property.
        value: A string attribute.
      """
            key = _messages.StringField(1)
            value = _messages.StringField(2)
        additionalProperties = _messages.MessageField('AdditionalProperty', 1, repeated=True)
    labels = _messages.MessageField('LabelsValue', 1)
    parentOperationId = _messages.StringField(2)
    requestId = _messages.StringField(3)