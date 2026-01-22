from __future__ import absolute_import
from apitools.base.protorpclite import messages as _messages
from apitools.base.py import encoding
from apitools.base.py import extra_types
class DerivedSource(_messages.Message):
    """Specification of one of the bundles produced as a result of splitting a
  Source (e.g. when executing a SourceSplitRequest, or when splitting an
  active task using WorkItemStatus.dynamic_source_split), relative to the
  source being split.

  Enums:
    DerivationModeValueValuesEnum: What source to base the produced source on
      (if any).

  Fields:
    derivationMode: What source to base the produced source on (if any).
    source: Specification of the source.
  """

    class DerivationModeValueValuesEnum(_messages.Enum):
        """What source to base the produced source on (if any).

    Values:
      SOURCE_DERIVATION_MODE_UNKNOWN: The source derivation is unknown, or
        unspecified.
      SOURCE_DERIVATION_MODE_INDEPENDENT: Produce a completely independent
        Source with no base.
      SOURCE_DERIVATION_MODE_CHILD_OF_CURRENT: Produce a Source based on the
        Source being split.
      SOURCE_DERIVATION_MODE_SIBLING_OF_CURRENT: Produce a Source based on the
        base of the Source being split.
    """
        SOURCE_DERIVATION_MODE_UNKNOWN = 0
        SOURCE_DERIVATION_MODE_INDEPENDENT = 1
        SOURCE_DERIVATION_MODE_CHILD_OF_CURRENT = 2
        SOURCE_DERIVATION_MODE_SIBLING_OF_CURRENT = 3
    derivationMode = _messages.EnumField('DerivationModeValueValuesEnum', 1)
    source = _messages.MessageField('Source', 2)