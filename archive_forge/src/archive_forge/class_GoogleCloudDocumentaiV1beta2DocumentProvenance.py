from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleCloudDocumentaiV1beta2DocumentProvenance(_messages.Message):
    """Structure to identify provenance relationships between annotations in
  different revisions.

  Enums:
    TypeValueValuesEnum: The type of provenance operation.

  Fields:
    id: The Id of this operation. Needs to be unique within the scope of the
      revision.
    parents: References to the original elements that are replaced.
    revision: The index of the revision that produced this element.
    type: The type of provenance operation.
  """

    class TypeValueValuesEnum(_messages.Enum):
        """The type of provenance operation.

    Values:
      OPERATION_TYPE_UNSPECIFIED: Operation type unspecified. If no operation
        is specified a provenance entry is simply used to match against a
        `parent`.
      ADD: Add an element.
      REMOVE: Remove an element identified by `parent`.
      UPDATE: Updates any fields within the given provenance scope of the
        message. It overwrites the fields rather than replacing them. Use this
        when you want to update a field value of an entity without also
        updating all the child properties.
      REPLACE: Currently unused. Replace an element identified by `parent`.
      EVAL_REQUESTED: Deprecated. Request human review for the element
        identified by `parent`.
      EVAL_APPROVED: Deprecated. Element is reviewed and approved at human
        review, confidence will be set to 1.0.
      EVAL_SKIPPED: Deprecated. Element is skipped in the validation process.
    """
        OPERATION_TYPE_UNSPECIFIED = 0
        ADD = 1
        REMOVE = 2
        UPDATE = 3
        REPLACE = 4
        EVAL_REQUESTED = 5
        EVAL_APPROVED = 6
        EVAL_SKIPPED = 7
    id = _messages.IntegerField(1, variant=_messages.Variant.INT32)
    parents = _messages.MessageField('GoogleCloudDocumentaiV1beta2DocumentProvenanceParent', 2, repeated=True)
    revision = _messages.IntegerField(3, variant=_messages.Variant.INT32)
    type = _messages.EnumField('TypeValueValuesEnum', 4)