from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class GoogleFirestoreAdminV1beta2IndexConfig(_messages.Message):
    """The index configuration for this field.

  Fields:
    ancestorField: Output only. Specifies the resource name of the `Field`
      from which this field's index configuration is set (when
      `uses_ancestor_config` is true), or from which it *would* be set if this
      field had no index configuration (when `uses_ancestor_config` is false).
    indexes: The indexes supported for this field.
    reverting: Output only When true, the `Field`'s index configuration is in
      the process of being reverted. Once complete, the index config will
      transition to the same state as the field specified by `ancestor_field`,
      at which point `uses_ancestor_config` will be `true` and `reverting`
      will be `false`.
    usesAncestorConfig: Output only. When true, the `Field`'s index
      configuration is set from the configuration specified by the
      `ancestor_field`. When false, the `Field`'s index configuration is
      defined explicitly.
  """
    ancestorField = _messages.StringField(1)
    indexes = _messages.MessageField('GoogleFirestoreAdminV1beta2Index', 2, repeated=True)
    reverting = _messages.BooleanField(3)
    usesAncestorConfig = _messages.BooleanField(4)